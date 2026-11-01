from __future__ import annotations
import asyncio
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw
from jsonschema import Draft202012Validator

from plot_reader import scatter_beta as beta, marker_fitting as fitting, jev_vision as jev
from plot_reader.luna_runtime import LunaRunner


def test_worker_forks_keep_frozen_profile_and_independent_state(monkeypatch):
    async def emit(event):pass
    monkeypatch.setenv('PLOT_READER_AGENT','codex')
    parent=LunaRunner(emit,effort='xhigh');parent.structured_output={'old':True}
    monkeypatch.setenv('PLOT_READER_AGENT','claude')
    a,b=parent.fork(),parent.fork()
    assert a.profile==b.profile==parent.profile
    assert a.profile['model']=='gpt-6-luna' and a.profile['effort']=='xhigh'
    assert a.structured_output is None and b.structured_output is None
    a.structured_output={'new':True}
    assert parent.structured_output=={'old':True} and b.structured_output is None
    asyncio.run(parent.release(a));asyncio.run(parent.release(b))
    assert not parent.children


def test_parent_cancellation_reaches_all_children(monkeypatch):
    stopped=[]
    class Process:
        returncode=None
        def __init__(self,pid):self.pid=pid
        async def wait(self):self.returncode=0
    async def emit(event):pass
    parent=LunaRunner(emit);a,b=parent.fork(),parent.fork()
    a.process=Process(111);b.process=Process(222)
    monkeypatch.setattr('plot_reader.luna_runtime.os.killpg',lambda pid,sig:stopped.append(pid))
    asyncio.run(parent.cancel())
    assert sorted(stopped)==[111,222]


def test_setup_schema_is_compact_and_rejects_missing_axis_checks():
    schema=beta.setup_schema()
    assert 'selections' not in schema['properties']
    assert schema['properties']['clarifications']['maxItems']==0
    axis=schema['properties']['x_axis']['anyOf'][0]
    assert axis['properties']['anchors']['maxItems']==2
    assert axis['properties']['checks']['minItems']==1
    Draft202012Validator.check_schema(schema)


def test_candidate_grouping_retains_every_seed_and_excluded_regions(tmp_path):
    source=tmp_path/'source.png';im=Image.new('RGB',(200,150),'white');draw=ImageDraw.Draw(im)
    draw.line((10,50,190,50),fill='black',width=2);draw.rectangle((77,47,83,53),outline='black');im.save(source)
    candidates=[{'id':f'M{i}','pixel':[x,50],'detector':kind,'offset_limit':3,'score':.6}
                for i,(x,kind) in enumerate([(80,'bounded contour'),(81,'ink corner'),(40,'ink corner'),(170,'ink corner')])]
    original=copy.deepcopy(candidates)
    primary,reserve=fitting.prepare_candidates(source,candidates,{'plot_bounds':[0,0,200,150],'exclude_regions':[[160,0,200,150]]})
    assert candidates==original
    assert {p['id'] for p in primary+reserve}=={'M0','M1','M2'}
    assert len(primary)+len(reserve)==3
    assert any(p.get('representative_id')=='M0' for p in reserve)
    assert fitting.line_evidence(np.asarray(im.convert('L')),[40,50],6)['line_only']
    assert not fitting.line_evidence(np.asarray(im.convert('L')),[80,50],6)['line_only']


def test_batches_have_unique_ownership_and_keep_native_coordinates(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(600,400),'white').save(source)
    points=[{'id':f'M{i}','pixel':[10+(i%100)*5,20+(i//100)*40],'offset_limit':3,'detector':'test'} for i in range(210)]
    batches=beta.review_batches(source,points[:140],points[140:],tmp_path/'batches')
    all_points=[p for b in batches for p in b['candidates']]
    assert len(batches)==6
    assert len({p['id'] for p in all_points})==len(points)
    assert {p['id']:p['pixel'] for p in all_points}=={p['id']:p['pixel'] for p in points}
    assert all(Path(b['catalogue_path']).is_file() for b in batches)


def test_reconciliation_retains_cross_series_overlap_and_flags_duplicate():
    candidates=[{'id':'A','pixel':[30,30]},{'id':'B','pixel':[31,30]}]
    def item(cid,sid):return {'candidate_id':cid,'series_id':sid,'verdict':'confirmed','offset':[0,0],'reason':'source'}
    records=[{'items':[item('A','S0'),item('B','S0'),item('B','S1')]}]
    merged,dropped=beta.reconcile(records,candidates,['square','cross'])
    assert len(merged[0]['items'])==2 and len(dropped)==1
    assert merged[0]['items'][0]['verdict']=='uncertain'


def test_jev_answers_must_account_for_exact_ids_and_finite_scores():
    body={'answers':{'M1':{'type':'choice','choice':'S0','probabilities':{'S0':.8,'ambiguous':.2}}}}
    assert jev.validate_answers(body,['M1'],{'S0','ambiguous'})['M1']['choice']=='S0'
    with pytest.raises(ValueError):jev.validate_answers(body,['M2'],{'S0','ambiguous'})
    body['answers']['M1']['probabilities']['S0']=float('nan')
    with pytest.raises(ValueError):jev.validate_answers(body,['M1'],{'S0','ambiguous'})


def test_jev_missing_key_falls_back_without_network(tmp_path,monkeypatch):
    monkeypatch.delenv('TYPESAFE_API_KEY',raising=False)
    report=asyncio.run(jev.classify_candidates(tmp_path/'not-needed.png',[],[],tmp_path/'jev',mode='assist'))
    assert report['status']=='unavailable' and report['answers']=={}
    assert 'apikey_' not in (tmp_path/'jev/report.json').read_text()


def test_beta_reviews_overlap_without_sharing_runner_and_coverage_is_explicit(tmp_path,monkeypatch):
    source=tmp_path/'source.png';Image.new('RGB',(200,200),'white').save(source)
    points=[{'id':f'M{i}','pixel':[20+i*40,60],'detector':'contour','offset_limit':0} for i in range(4)]
    monkeypatch.setattr(beta,'marker_candidates',lambda s:points)
    async def no_template(*args):return None
    monkeypatch.setattr(beta,'request_template_evidence',no_template)
    monkeypatch.setattr(beta,'prepare_candidates',lambda *args:(points,[]))
    monkeypatch.setattr(beta,'review_batches',lambda *args:[{'kind':'primary','bounds':[0,0,200,200], 'source_path':str(source),'catalogue_path':str(source),'candidates':[p]} for p in points])
    state={'active':0,'peak':0,'children':set()}
    class Child:
        async def call(self,**kwargs):
            state['active']+=1;state['peak']=max(state['peak'],state['active'])
            await asyncio.sleep(.02)
            state['active']-=1
            cid=kwargs['schema']['properties']['items']['items']['properties']['candidate_id']['enum'][0]
            return {'items':[{'candidate_id':cid,'series_id':'S0','marker_visible':True,'center_supported':True,'verdict':'confirmed','offset':[0,0],'reason':'square'}],'notes':[]}
    class Runner:
        async def emit(self,event):pass
        def fork(self):
            c=Child();state['children'].add(c);return c
        async def release(self,child):state['children'].remove(child)
        async def call(self,**kwargs):return {'coverage_supported':False,'issues':['A boundary marker is not supported.']}
    plan={'selections':[],'marker_series':[{'name':'A','shape':'square','guide':'squares'}],
          'marker_layout':{'plot_bounds':[0,0,200,200],'exclude_regions':[],'series_guidance':'square'}}
    result=asyncio.run(beta.ground_scatter_beta(Runner(),plan,source,tmp_path,{'targets':'all points'},concurrency=2))
    assert state['peak']==2 and not state['children']
    assert len(result['selections'])==4
    assert result['inventory_complete'] is False
    assert result['unresolved_inventory']==['A boundary marker is not supported.']


def test_joint_marker_fit_requires_ink_beyond_line_and_recovers_offset():
    im=Image.new('L',(400,300),255);d=ImageDraw.Draw(im)
    d.line((5,100,395,100),fill=0,width=1)
    d.line((196,100,204,100),fill=0,width=1);d.line((200,96,200,104),fill=0,width=1)
    gray=np.asarray(im)
    positive=fitting.fit_marker_plus_line(gray,[199,100],'plus')
    negative=fitting.fit_marker_plus_line(gray,[100,100],'plus')
    assert positive['supported'] and abs(positive['pixel'][0]-200)<=1
    assert not negative['supported']


def test_browser_measures_large_inventory_without_truncation(tmp_path,monkeypatch):
    from plot_reader import browser_measurement as measurement
    calls=[]
    def engine(session,x,selections,y,reference,baseline,proposals):
        assert len(selections)<=100
        calls.append(len(selections))
        return {'points':[{'label':s['label'],'pixel':s['pixel'],'x':s['pixel'][0]} for s in selections]}
    monkeypatch.setattr(measurement.core,'measure_plot',engine)
    points=[{'label':f'p{i}','pixel':[i,20]} for i in range(205)]
    axis={'anchors':[{'pixel':0,'value':0},{'pixel':300,'value':300}],'checks':[{'pixel':150,'value':150}]}
    result=measurement.measure_selected('session',axis,points)
    assert calls==[100,100,5]
    assert [p['label'] for p in result['points']]==[p['label'] for p in points]


def test_jev_http_failure_is_advisory_and_never_serializes_key(tmp_path,monkeypatch):
    import httpx
    secret='test-key-not-for-output'
    monkeypatch.setenv('TYPESAFE_API_KEY',secret)
    source=tmp_path/'source.png';Image.new('RGB',(300,300),'white').save(source)
    class Client:
        def __init__(self,**kwargs):pass
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def post(self,url,**kwargs):
            assert kwargs['headers']['Authorization']=='Bearer '+secret
            request=httpx.Request('POST',url)
            return httpx.Response(429,request=request,json={'secret':secret})
    monkeypatch.setattr(jev.httpx,'AsyncClient',Client)
    folder=tmp_path/'jev'
    result=asyncio.run(jev.classify_candidates(source,[{'id':'M1','pixel':[100,100]}],[{'name':'A','shape':'square'}],folder,mode='assist'))
    assert result['failed_batches']==1 and not result['answers']
    assert all(secret not in p.read_text() for p in folder.glob('*.json'))


def test_local_acceptance_blocks_hallucinated_line_marker(tmp_path):
    source=tmp_path/'source.png';im=Image.new('RGB',(400,300),'white');draw=ImageDraw.Draw(im)
    draw.line((10,100,390,100),fill='black');draw.line((200,96,200,104),fill='black');im.save(source)
    candidates=[{'id':'M1','pixel':[100,100],'detector':'ink corner','offset_limit':3},
                {'id':'M2','pixel':[200,100],'detector':'ink corner','offset_limit':3}]
    records=[{'items':[{'candidate_id':p['id'],'series_id':'S0','offset':[0,0],'verdict':'confirmed','reason':'model claims plus'} for p in candidates]}]
    valid,rejected=fitting.validate_marker_selections(source,candidates,records,[{'shape':'plus'}],{'plot_bounds':[10,10,390,290]})
    assert [p['candidate_id'] for p in valid[0]['items']]==['M2']
    assert [p['candidate_id'] for p in rejected]==['M1']


def test_beta_rejected_points_and_template_centers_do_not_move(tmp_path):
    source=tmp_path/'source.png';im=Image.new('RGB',(400,300),'white');draw=ImageDraw.Draw(im)
    draw.line((10,100,390,100),fill='black');draw.line((200,96,200,104),fill='black');im.save(source)
    p={'id':'K1','pixel':[200.25,100.25],'detector':'source template correlation','offset_limit':0,'template_series':'S0'}
    rec={'candidate_id':'K1','series_id':'S0','offset':[0,0],'verdict':'confirmed','reason':'plus'}
    valid,rejected=fitting.validate_marker_selections(source,[p],[{'items':[rec]}],[{'shape':'plus'}],{'plot_bounds':[10,10,390,290]})
    assert not rejected and valid[0]['items'][0]['offset']==[0,0]
    assert p['pixel']==[200.25,100.25]


def test_strip_ocr_rejects_inferred_unprinted_check(tmp_path,monkeypatch):
    source=tmp_path/'source.png';Image.new('RGB',(500,400),'white').save(source)
    ocr={'text':['0','200','400','600'],'conf':[95]*4,'left':[10,300,600,900],
         'top':[0]*4,'width':[20]*4,'height':[30]*4}
    monkeypatch.setattr(beta.pytesseract,'image_to_data',lambda *a,**k:ocr)
    plan={'marker_layout':{'plot_bounds':[50,10,450,300]},'x_axis':{'scale':'linear','checks':[{'value':300,'pixel':250}]},'y_axis':None}
    evidence,errors=beta.printed_tick_checks(source,plan)
    assert len(errors)==1 and 'no matching printed label' in errors[0]
    plan['x_axis']['checks'][0]['value']=200
    assert beta.printed_tick_checks(source,plan)[1]==[]


def test_jev_malformed_answer_fails_validation_not_attribute_error():
    for body in [[],{'answers':{'M1':None}}]:
        with pytest.raises(ValueError):jev.validate_answers(body,['M1'],{'non_marker'})
