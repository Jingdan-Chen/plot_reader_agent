import asyncio
import io
import json
import re
from pathlib import Path

import pytest
from PIL import Image
from starlette.testclient import TestClient
from plot_reader import browser,luna_runtime as runtime
from plot_reader.timing import AnalysisTiming, current_timing

SCHEMA={'type':'object','properties':{'ready':{'type':'boolean'}},'required':['ready'],'additionalProperties':False}


def fake_cli(monkeypatch,provider,result):
    calls=[]
    monkeypatch.setenv('PLOT_READER_AGENT',provider)
    monkeypatch.setattr(runtime.shutil,'which',lambda _: '/mock/'+provider)
    monkeypatch.setattr(runtime,'check_claude_cli',lambda _:None)
    class Input:
        def write(self,data):calls[-1]['input']=data
        async def drain(self):pass
        def close(self):pass
    class Process:
        returncode=0
        def __init__(self,args):
            self.stdin=Input();self.stdout=asyncio.StreamReader();self.stderr=asyncio.StreamReader()
            event={'type':'result','structured_output':result} if provider=='claude' else {'type':'thread.started','thread_id':'mock'}
            self.stdout.feed_data((json.dumps(event)+'\n').encode());self.stdout.feed_eof();self.stderr.feed_eof()
            if provider=='codex':Path(args[args.index('-o')+1]).write_text(json.dumps(result))
        async def wait(self):return 0
    async def spawn(*args,**kwargs):
        calls.append({'args':args,'options':kwargs});return Process(args)
    monkeypatch.setattr(runtime.asyncio,'create_subprocess_exec',spawn)
    return calls


@pytest.mark.parametrize('provider',['codex','claude'])
def test_provider_receives_image_exact_profile_and_validates_output(tmp_path,monkeypatch,provider):
    calls=fake_cli(monkeypatch,provider,{'ready':True});events=[]
    monkeypatch.setenv('TYPESAFE_API_KEY','not-forwarded');monkeypatch.setenv('OPENAI_API_KEY','not-forwarded');monkeypatch.setenv('CLAUDECODE','1')
    image=tmp_path/'input.png';Image.new('RGB',(96,96),'white').save(image)
    async def emit(event):events.append(event)
    timing=AnalysisTiming(tmp_path/'timing')
    token=current_timing.set(timing)
    try:
        result=asyncio.run(runtime.LunaRunner(emit).call(prompt='Image task',schema=SCHEMA,images=[str(image)],workdir=str(tmp_path/'selected-plan')))
    finally:
        current_timing.reset(token)
    report=timing.finish()
    assert len(report['model_calls'])==1
    assert report['model_calls'][0]['provider']==provider
    assert report['model_calls'][0]['name']=='selected-plan'
    assert report['summary']['model_wait_seconds']>0
    if provider=='claude':assert len(report['local_python_calls'])==2
    assert result=={'ready':True}
    args=calls[0]['args'];environment=calls[0]['options']['env']
    assert 'OPENAI_API_KEY' not in environment and 'CLAUDECODE' not in environment and 'TYPESAFE_API_KEY' not in environment
    assert events[0]['message']=='Locating the requested peaks and points'
    if provider=='codex':
        assert 'service_tier=priority' in args and 'model_reasoning_effort=medium' in args and 'gpt-6-luna' in args
        assert '--ignore-user-config' in args and str(image) in args
    else:
        assert args[args.index('--model')+1]=='claude-haiku-5-5'
        assert args[args.index('--effort')+1]=='medium'
        assert args[args.index('--tools')+1]=='' and '--restricted' in args
        content=json.loads(calls[0]['input'])['message']['content']
        assert content[1]['source']['media_type']=='image/png' and content[1]['source']['data']
    assert json.loads((tmp_path/'selected-plan/inference.json').read_text())['provider']==provider


@pytest.mark.parametrize('provider,choice,native',[
    ('codex','light','low'),('codex','medium','medium'),('codex','high','high'),('codex','xhigh','xhigh'),
    ('claude','low','low'),('claude','medium','medium'),('claude','high','high'),('claude','extra','xhigh')])
def test_run_effort_is_applied_to_every_cli_call_and_recorded(tmp_path,monkeypatch,provider,choice,native):
    calls=fake_cli(monkeypatch,provider,{'ready':True})
    async def emit(event):pass
    async def run():
        runner=runtime.LunaRunner(emit,effort=choice)
        for stage in ['selected-plan','explanation']:
            await runner.call(prompt='Task',schema=SCHEMA,images=[],workdir=str(tmp_path/stage))
            # Changing defaults cannot alter this already-created runner.
            monkeypatch.setenv('PLOT_READER_AGENT','claude' if provider=='codex' else 'codex')
    asyncio.run(run())
    assert len(calls)==2
    for call in calls:
        args=call['args']
        if provider=='codex':
            assert f'model_reasoning_effort={native}' in args
            assert 'gpt-6-luna' in args and 'service_tier=priority' in args
        else:
            assert args[args.index('--effort')+1]==native
            assert args[args.index('--model')+1]=='claude-haiku-5-5'
    for stage in ['selected-plan','explanation']:
        saved=json.loads((tmp_path/stage/'inference.json').read_text())
        assert saved['effort_choice']==choice and saved['effort']==native
        assert saved['provider']==provider


@pytest.mark.parametrize('provider,expected',[('codex',['light','medium','high','xhigh']),('claude',['low','medium','high','extra'])])
def test_portal_effort_options_match_provider_with_medium_selected(tmp_path,monkeypatch,provider,expected):
    monkeypatch.setenv('PLOT_READER_AGENT',provider)
    monkeypatch.setattr(browser,'DATA_ROOT',tmp_path)
    monkeypatch.setattr(browser.app.state,'token','test')
    with TestClient(browser.app,base_url='http://127.0.0.1') as client:
        page=client.get('/?token=test').text
    select=re.search(r'<select id="model-effort"[^>]*>(.*?)</select>',page,re.S).group(1)
    assert re.findall(r'<option value="([^"]+)"',select)==expected
    assert re.search(r'<option value="medium"[^>]* selected>',select)
    assert '__MODEL_EFFORT_OPTIONS__' not in page


def test_saved_run_shows_its_original_inference_profile(tmp_path,monkeypatch):
    monkeypatch.setattr(browser,'DATA_ROOT',tmp_path)
    monkeypatch.setattr(browser.app.state,'token','test')
    job_id='d'*32;folder=tmp_path/'jobs'/job_id
    browser._json(folder/'result.json',{'rows':[]})
    profile={'provider':'codex','model':'gpt-6-luna','effort':'high','label':'LUNA 6 · high · Fast'}
    broken=folder/'worker'/'aaa'/'inference.json';broken.parent.mkdir(parents=True);broken.write_text('incomplete')
    browser._json(folder/'worker'/'selected-plan'/'inference.json',profile)
    with TestClient(browser.app,base_url='http://127.0.0.1') as client:
        saved=client.get(f'/api/jobs/{job_id}/result',headers={'x-plot-reader-token':'test'}).json()
    assert saved['inference_profile']==profile
    assert 'inference_profile' not in json.loads((folder/'result.json').read_text())


def test_schema_failure_does_not_claim_completed(tmp_path,monkeypatch):
    fake_cli(monkeypatch,'claude',{'ready':'not a boolean'});events=[]
    async def emit(event):events.append(event)
    with pytest.raises(runtime.LunaError,match='violates'):
        asyncio.run(runtime.LunaRunner(emit).call(prompt='Task',schema=SCHEMA,images=[],workdir=str(tmp_path)))
    assert not any(event['type']=='luna_completed' for event in events)


def test_old_claude_cli_fails_before_model_execution(monkeypatch):
    runtime.check_claude_cli.cache_clear()
    class Help:
        returncode=0;stdout='old CLI --model --print'
    monkeypatch.setattr(runtime.subprocess,'run',lambda *args,**kwargs:Help())
    with pytest.raises(runtime.LunaError,match='too old'):
        runtime.check_claude_cli('/old/claude')


def test_latest_action_survives_reconnect_without_model_reasoning(tmp_path,monkeypatch):
    monkeypatch.setattr(browser,'DATA_ROOT',tmp_path)
    monkeypatch.setattr(browser.app.state,'token','test')
    job_id='a'*32;browser._json(tmp_path/'jobs'/job_id/'job.json',{'id':job_id,'status':'running'})
    with TestClient(browser.app,base_url='http://127.0.0.1') as client:
        async def emit():
            await browser._emit(browser.app,job_id,'luna_started','Checking source markers')
            await browser._emit(browser.app,job_id,'luna_event','private content must not become the activity label')
        asyncio.run(emit())
        saved=client.get(f'/api/jobs/{job_id}',headers={'x-plot-reader-token':'test'}).json()
        assert saved['latest_action']['message']=='Checking source markers'
        assert saved['latest_action']['at']


def test_claude_portal_displays_actual_provider(tmp_path,monkeypatch):
    monkeypatch.setenv('PLOT_READER_AGENT','claude');monkeypatch.setattr(browser,'DATA_ROOT',tmp_path)
    monkeypatch.setattr(browser.app.state,'token','test')
    with TestClient(browser.app,base_url='http://127.0.0.1') as client:
        response=client.get('/?token=test')
        assert 'Haiku 5.5 · medium' in response.text and 'Haiku · latest action' in response.text
        assert '__ASSISTANT_' not in response.text
