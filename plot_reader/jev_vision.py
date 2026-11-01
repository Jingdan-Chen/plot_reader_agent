"""Explicit, optional JEV image classification; never provides measurement pixels."""
from __future__ import annotations
import asyncio
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import uuid

import httpx
from PIL import Image, ImageDraw, ImageFont
from .timing import timing_span

MODEL = 'jev-1.14.0'
ENDPOINT = 'https://api.typesafe.ai/v1/systemone'


def candidate_sheet(source: Path, candidates: list[dict]) -> Image.Image:
    image=Image.open(source).convert('RGB');cell=280;columns=min(4,len(candidates))
    sheet=Image.new('RGB',(max(280,columns*cell),max(280,((len(candidates)+columns-1)//columns)*cell)), 'white')
    draw=ImageDraw.Draw(sheet);font=ImageFont.load_default(size=20)
    radius=max(8,round(max(image.size)/1432*18))
    for i,p in enumerate(candidates):
        x,y=p['pixel'];patch=image.crop((round(x)-radius,round(y)-radius,round(x)+radius+1,round(y)+radius+1))
        patch=patch.resize((248,248),Image.Resampling.NEAREST)
        u=(i%columns)*cell+16;v=(i//columns)*cell+30
        sheet.paste(patch,(u,v));draw.text((u,v-26),p['id'],fill='black',font=font)
    return sheet


def validate_answers(body: dict, ids: list[str], options: set[str]) -> dict:
    if not isinstance(body, dict):raise ValueError('Invalid JEV response')
    answers=body.get('answers')
    if not isinstance(answers,dict) or set(answers)!=set(ids):raise ValueError('Missing or foreign JEV answers')
    result={}
    for key in ids:
        item=answers[key]
        if not isinstance(item, dict):raise ValueError('Invalid JEV answer')
        choice=item.get('choice');probabilities=item.get('probabilities')
        if item.get('type')!='choice' or choice not in options or not isinstance(probabilities,dict) or set(probabilities)!=options:
            raise ValueError('Invalid JEV classification')
        values=list(probabilities.values())
        if any(type(v) not in (float,int) or not 0<=v<=1 for v in values) or abs(sum(values)-1)>.02:
            raise ValueError('Invalid JEV probabilities')
        result[key]={'choice':choice,'probabilities':probabilities,
                     'note':'Experimental model scores; not calibrated marker confidence.'}
    return result


async def classify_candidates(source: Path, candidates: list[dict], series: list[dict], folder: Path,
                              *, mode: str = 'shadow', emit=None) -> dict:
    if mode not in {'shadow','assist'}:return {'status':'off','answers':{}}
    folder.mkdir(parents=True,exist_ok=True)
    key=os.environ.get('TYPESAFE_API_KEY')
    if not key:
        report={'status':'unavailable','mode':mode,'model':MODEL,'reason':'TYPESAFE_API_KEY is not configured; selected agent reviews all candidates.','answers':{}}
        (folder/'report.json').write_text(json.dumps(report,indent=2));return report
    criteria={f'S{i}':f"A visible {s['shape']} marker belonging to {s['name']}." for i,s in enumerate(series)}
    criteria.update(non_marker='Only fitted-line ink, axis, text, or blank space; no marker at patch center.',
                    ambiguous='A marker may exist but its shape, series, or center cannot be distinguished.')
    batches=[candidates[i:i+16] for i in range(0,len(candidates),16)]
    sem=asyncio.Semaphore(2)
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        async def one(index,batch):
            async with sem:
                sheet=candidate_sheet(source,batch);buffer=io.BytesIO();sheet.save(buffer,format='PNG')
                ref='image-'+uuid.uuid4().hex
                payload={'model':MODEL,'state':{'candidate_sheet':ref,'guide':'Each labeled patch is centered on a source-derived proposal. Classify the visible marker shape at its center. Curves and axes are not markers.'},
                    'assets':{ref:{'type':'image','format':'png','data_base64':base64.b64encode(buffer.getvalue()).decode()}},
                    'questions':{p['id']:{'type':'choice','instructions':f"Classify the source patch labeled {p['id']} in candidate_sheet. Do not infer from curve position or expected counts.",'criteria':criteria} for p in batch}}
                signature=hashlib.sha256(buffer.getvalue()+json.dumps(criteria,sort_keys=True).encode()+MODEL.encode()).hexdigest()
                try:
                    with timing_span('model_wait',f'jev-candidates-{index}',provider='typesafe',model=MODEL,mode=mode):
                        response=await client.post(ENDPOINT,json=payload,headers={'Authorization':'Bearer '+key,'TypeSafe-Danger-Zone-Images':'true'})
                        response.raise_for_status();body=response.json()
                        answers=validate_answers(body,[p['id'] for p in batch],set(criteria))
                    record={'status':'complete','source_hash':signature,'model':body.get('model'),'answers':answers,'usage':body.get('usage',{})}
                except (httpx.HTTPError,ValueError,TypeError,KeyError) as exc:
                    # Never persist request headers, response bodies, or exception text.
                    record={'status':'failed','error_type':type(exc).__name__,
                            'http_status':exc.response.status_code if isinstance(exc,httpx.HTTPStatusError) else None,'answers':{}}
                (folder/f'batch-{index}.json').write_text(json.dumps(record,indent=2,allow_nan=False))
                return record
        records=await asyncio.gather(*(one(i,b) for i,b in enumerate(batches)))
    answers={k:v for r in records for k,v in r['answers'].items()}
    report={'status':'complete' if all(r['status']=='complete' for r in records) else 'partial',
            'mode':mode,'model':MODEL,'batches':len(records),'answers':answers,
            'failed_batches':sum(r['status']!='complete' for r in records),
            'policy':'Advisory classification only. No coordinates, point acceptance, or deletion delegated to JEV.'}
    (folder/'report.json').write_text(json.dumps(report,indent=2))
    if emit:await emit({'type':'stage','stage':'coordinate_review','message':f"JEV classified {len(answers)} candidate patches; source review remains with the selected agent"})
    return report
