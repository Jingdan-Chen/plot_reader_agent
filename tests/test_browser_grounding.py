from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from starlette.testclient import TestClient

from plot_reader import browser, core
from plot_reader.browser_grounding import marker_candidates, bind_inventory, candidate_tiles


def test_small_markers_connected_to_fit_are_found_at_source_resolution(tmp_path):
    image=Image.new('RGB',(600,400),'white');d=ImageDraw.Draw(image)
    d.line((30,103,570,97),fill='black',width=1)
    d.rectangle((98,98,102,102),outline='black')
    d.line((198,98,202,102),fill='black');d.line((198,102,202,98),fill='black')
    d.line((298,100,302,100),fill='black');d.line((300,97,300,103),fill='black')
    d.polygon([(398,98),(402,98),(400,102)],outline='black',fill='white')
    for factor in (1,2):
        source=tmp_path/f'source-{factor}.png';image.resize((600*factor,400*factor)).save(source)
        candidates=marker_candidates(source)
        for x in (100,200,300,400):
            assert min(math.dist(p['pixel'],[x*factor,100*factor]) for p in candidates)<=3*factor
        assert not any(math.dist(p['pixel'],[500*factor,250*factor])<20 for p in candidates)


def test_binding_discards_drifting_seeds_and_keeps_missing_items_explicit():
    plan={'selections':[{'label':'one','series':'square','pixel':[30,180]},
                        {'label':'two','series':'square','pixel':[200,180]}]}
    before=copy.deepcopy(plan)
    candidates=[{'id':'M1','pixel':[30,30],'detector':'contour','offset_limit':3}]
    records=[{'items':[{'candidate_id':'M1','series_id':'S0','verdict':'confirmed','offset':[1,0],'reason':'visible square'}]}]
    result=bind_inventory(plan,candidates,records,['square'])
    assert plan==before
    assert result['selections'][0]['pixel']==[31,30]
    assert result['selections'][1]['pixel'] is None
    assert result['selections'][1]['status']=='unresolved'
    records[0]['items'][0]['offset']=[4,0]
    with pytest.raises(ValueError,match='local patch'):bind_inventory(plan,candidates,records,['square'])
    records[0]['items'][0]['candidate_id']='foreign'
    with pytest.raises(ValueError,match='Unknown'):bind_inventory(plan,candidates,records,['square'])


def test_catalogue_owns_each_candidate_once_and_excludes_legend(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(600,400),'white').save(source)
    candidates=[{'id':f'M{i}','pixel':[i*5+10,100],'detector':'test','offset_limit':3} for i in range(110)]
    plan={'marker_layout':{'plot_bounds':[0,0,600,400],'exclude_regions':[[200,50,300,200]]}}
    tiles=candidate_tiles(source,candidates,tmp_path/'tiles',plan)
    ids=[p['id'] for t in tiles for p in t['candidates']]
    assert len(ids)==len(set(ids))
    assert set(ids)=={p['id'] for p in candidates if not 200<=p['pixel'][0]<=300}


def test_explicit_correction_remeasures_new_revision_and_rejects_bad_pixels(tmp_path,monkeypatch):
    monkeypatch.setattr(browser,'DATA_ROOT',tmp_path/'browser')
    monkeypatch.setattr(core,'SESSION_ROOT',tmp_path/'sessions')
    job_id='a'*32;folder=browser._job_dir(job_id);folder.mkdir(parents=True)
    source=folder/'source-image.png';image=Image.new('RGB',(600,400),'white');ImageDraw.Draw(image).ellipse((295,95,305,105),outline='black');image.save(source)
    inspection=core.inspect_plot(str(source));browser._json(folder/'inspection.json',inspection)
    plan={'x_axis':{'anchors':[{'pixel':50,'value':0},{'pixel':550,'value':10}],'scale':'linear','unit':'min'},
          'y_axis':{'anchors':[{'pixel':350,'value':0},{'pixel':50,'value':1}],'scale':'linear'},
          'selections':[{'label':'A','series':'circle','pixel':[300,250],'mode':'exact'}]}
    measured=browser.measure_selected(inspection['session_id'],plan['x_axis'],plan['selections'],plan['y_axis'])
    data={'mode':'selected','normalization':False}
    for name,value in [('job',{'id':job_id,'status':'complete','engine':'full'}),('request',data),('plan',plan),('engine-result',measured),('result',{'source':{},'experimental':False})]:browser._json(folder/f'{name}.json',value)
    browser.app.state.token='test';browser.app.state.tasks={}
    with TestClient(browser.app,base_url='http://127.0.0.1') as client:
        headers={'x-plot-reader-token':'test'}
        url=f'/api/jobs/{job_id}/corrections'
        assert client.post(url,json={'edits':[{'index':0,'pixel':[600,100]}]},headers=headers).status_code==400
        assert len(list((browser.DATA_ROOT/'jobs').iterdir()))==1
        response=client.post(url,json={'edits':[{'index':0,'pixel':[300,100]}]},headers=headers)
        assert response.status_code==201,response.text
        new=browser._job_dir(response.json()['id']);result=browser._read(new/'result.json')
        assert result['rows'][0]['x']==pytest.approx(5)
        assert result['rows'][0]['x_unit']=='min'
        assert result['rows'][0]['y']==pytest.approx(250/300)
        assert browser._read(folder/'plan.json')['selections'][0]['pixel']==[300,250]
        assert browser._read(new/'corrections.json')['inference_used'] is False
        exported=list(csv.DictReader((new/'table.csv').open(encoding='utf-8-sig')))[0]
        assert float(exported['y'])==result['rows'][0]['y']
        assert client.get(f'/api/jobs/{response.json()["id"]}/artifact/source-image.png',headers=headers).status_code==200
        changed=client.post(f'/api/jobs/{response.json()["id"]}/corrections',headers=headers,json={
            'edits':[{'index':0,'omit':True}],
            'additions':[{'label':'B','series':'circle','pixel':[400,100],'uncertain':True,'note':'Close markers'}]})
        assert changed.status_code==201,changed.text
        revised=browser._read(browser._job_dir(changed.json()['id'])/'result.json')
        assert len(revised['rows'])==1
        assert revised['rows'][0]['x']==pytest.approx(7)
        assert revised['rows'][0]['status']=='needs_review'
        assert result['rows'][0]['x']==pytest.approx(5)


def test_reported_drift_fixture_has_local_candidates_far_from_bad_seeds(tmp_path):
    source=Path(__file__).parent/'fixtures/browser/conversion-markers.png'
    if not source.is_file():
        # The public release generates equivalent small-marker geometry rather
        # than distributing a user's uploaded scientific figure.
        source=tmp_path/'small-markers.png';image=Image.new('RGB',(580,250),'white');draw=ImageDraw.Draw(image)
        for bounds in [(227,155,231,159),(227,162,231,166),(185,75,189,79),(145,152,150,157)]:
            draw.rectangle(bounds,outline='black',width=1)
        image.save(source)
    candidates=marker_candidates(source)
    # Actual visible centers checked in enlarged unannotated source crops.
    # These include a 5-pixel square, a tiny hollow triangle and an open plus.
    for pixel in ([228.9,156.7],[228.5,163.5],[187,77],[147.5,154.5]):
        assert min(math.dist(p['pixel'],pixel) for p in candidates)<1.5
    nearest=min(candidates,key=lambda p:math.dist(p['pixel'],[228.9,156.7]))
    assert math.dist(nearest['pixel'],[230,205])>45  # previously retained wrong seed
    result=bind_inventory({'selections':[{'label':'square-5','series':'square','pixel':[230,205]}]},
        candidates,[{'items':[{'candidate_id':nearest['id'],'series_id':'S0','offset':[0,0],
                              'verdict':'confirmed','reason':'center of visible square'}]}],['square'])
    assert result['selections'][0]['pixel']==nearest['pixel']
    assert result['selections'][0]['label']=='square-5'


def test_malformed_local_inventory_gets_one_repair_without_reusing_bad_seed(tmp_path,monkeypatch):
    import asyncio
    from plot_reader import browser_grounding as grounding
    source=tmp_path/'source.png';Image.new('RGB',(100,100),'white').save(source)
    candidate={'id':'M1','pixel':[30,30],'detector':'contour','offset_limit':3}
    monkeypatch.setattr(grounding,'marker_candidates',lambda _: [candidate])
    monkeypatch.setattr(grounding,'candidate_tiles',lambda *args:[{'bounds':[0,0,100,100],
        'source_path':str(source),'catalogue_path':str(source),'candidates':[candidate]}])
    class Runner:
        def __init__(self):self.calls=[]
        async def emit(self,event):pass
        async def call(self,**kwargs):
            self.calls.append(kwargs)
            if len(self.calls)==1:return {'plot_bounds':[0,0,100,100],'exclude_regions':[],'reason':'plot','series_guidance':'square'}
            return {'items':[{'candidate_id':'not-a-candidate','series_id':'S0','verdict':'confirmed','offset':[0,0],'reason':'bad'}],'notes':[]}
    runner=Runner()
    result=asyncio.run(grounding.ground_scatter(runner,{'selections':[{'label':'A','series':'square','pixel':[30,80]}]},source,tmp_path,'all markers'))
    assert len(runner.calls)==3
    assert result['selections'][0]['status']=='unresolved'
    assert result['selections'][0]['pixel'] is None
    assert any('failed' in note for note in result['notes'])


def test_missing_point_does_not_shift_neighboring_labels():
    plan={'selections':[{'label':f'p{i}','series':'square','pixel':[x,100]} for i,x in enumerate([10,30,50,70])]}
    candidates=[{'id':f'M{i}','pixel':[x,100],'detector':'contour','offset_limit':3} for i,x in enumerate([10,50,70])]
    records=[{'items':[{'candidate_id':p['id'],'series_id':'S0','offset':[0,0],'verdict':'confirmed','reason':'square'} for p in candidates]}]
    result=bind_inventory(plan,candidates,records,['square'])
    by_label={p['label']:p for p in result['selections']}
    assert by_label['p1']['pixel'] is None
    assert by_label['p2']['pixel']==[50,100]
    assert by_label['p3']['pixel']==[70,100]
