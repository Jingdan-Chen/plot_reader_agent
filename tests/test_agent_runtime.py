import asyncio
import io
import json
from pathlib import Path

import pytest
from PIL import Image
from starlette.testclient import TestClient
from plot_reader import browser,luna_runtime as runtime

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
    monkeypatch.setenv('OPENAI_API_KEY','not-forwarded');monkeypatch.setenv('CLAUDECODE','1')
    image=tmp_path/'input.png';Image.new('RGB',(96,96),'white').save(image)
    async def emit(event):events.append(event)
    result=asyncio.run(runtime.LunaRunner(emit).call(prompt='Image task',schema=SCHEMA,images=[str(image)],workdir=str(tmp_path/'selected-plan')))
    assert result=={'ready':True}
    args=calls[0]['args'];environment=calls[0]['options']['env']
    assert 'OPENAI_API_KEY' not in environment and 'CLAUDECODE' not in environment
    assert events[0]['message']=='Locating the requested peaks and points'
    if provider=='codex':
        assert 'service_tier=priority' in args and 'model_reasoning_effort=high' in args and 'gpt-6-luna' in args
        assert '--ignore-user-config' in args and str(image) in args
    else:
        assert args[args.index('--model')+1]=='claude-haiku-5-5'
        assert args[args.index('--effort')+1]=='medium'
        assert args[args.index('--tools')+1]=='' and '--restricted' in args
        content=json.loads(calls[0]['input'])['message']['content']
        assert content[1]['source']['media_type']=='image/png' and content[1]['source']['data']
    assert json.loads((tmp_path/'selected-plan/inference.json').read_text())['provider']==provider


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
