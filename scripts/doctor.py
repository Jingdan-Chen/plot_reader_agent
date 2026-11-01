#!/usr/bin/env python3
"""Check a clone's local runtime; --smoke-test makes one real model request."""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent',choices=['codex','claude'],default=os.environ.get('PLOT_READER_AGENT','codex'))
    parser.add_argument('--smoke-test',action='store_true',help='Make one billable/included-usage model request with a synthetic image')
    args=parser.parse_args();os.environ['PLOT_READER_AGENT']=args.agent
    problems=[]
    def check(label,okay,detail):
        print(f"{'OK' if okay else 'MISSING'}  {label}: {detail}")
        if not okay:problems.append(label)
    check('Python',sys.version_info>=(3,11),sys.version.split()[0])
    for package in ['Pillow','numpy','opencv-python-headless','scipy','pytesseract','mcp','starlette','uvicorn','python-multipart','requests','jsonschema']:
        try:check(package,True,importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:check(package,False,'Run uv sync --extra test')
    for program in ['tesseract','pdfinfo','pdfseparate','pdfunite','pdftoppm']:
        executable=shutil.which(program);check(program,bool(executable),executable or 'Install Tesseract/Poppler; see README')
    executable=shutil.which(args.agent);check('Inference CLI',bool(executable),executable or f'Install and sign in to {args.agent}')
    if problems:return 1
    from plot_reader.luna_runtime import assistant_profile,check_claude_cli,LunaRunner,LunaError
    from plot_reader.pdf_figures import skill_directory
    print('Inference profile:',assistant_profile()['label'])
    if args.agent=='claude':
        try:check_claude_cli(executable)
        except Exception as exc:check('Claude CLI capabilities',False,str(exc));return 1
    try:
        scripts=skill_directory();check('PDF skill',True,str(scripts))
        selected=ROOT/'plot_reader/checkpoints/selection.json';metadata=json.loads(selected.read_text())
        checkpoint=selected.parent/'snapshots'/metadata['snapshot'];manifest=checkpoint/'snapshot.json'
        if hashlib.sha256(manifest.read_bytes()).hexdigest()!=metadata['snapshot_sha256']:raise ValueError('manifest mismatch')
        for relative,expected in json.loads(manifest.read_text()).items():
            if hashlib.sha256((checkpoint/relative).read_bytes()).hexdigest()!=expected:raise ValueError(f'checkpoint mismatch: {relative}')
        check('Pinned curve engine',True,metadata['snapshot'])
    except Exception as exc:check('Packaged assets',False,str(exc));return 1
    print('PDF API token:', 'present (value hidden)' if os.environ.get('PADDLEOCR_API_TOKEN') else 'not set; image uploads work, PDF discovery needs it')
    if args.smoke_test:
        from PIL import Image,ImageDraw
        async def run():
            with tempfile.TemporaryDirectory(prefix='plot-reader-smoke-') as temporary:
                image=Path(temporary)/'synthetic.png';im=Image.new('RGB',(96,96),'white');ImageDraw.Draw(im).rectangle((30,30,66,66),fill='black');im.save(image)
                async def emit(event):
                    if event.get('message'):print(event['message'])
                result=await LunaRunner(emit,timeout=180).call(prompt='Inspect the attached image. Return ready=true if it contains a black square on a white background.',
                    schema={'type':'object','properties':{'ready':{'type':'boolean'}},'required':['ready'],'additionalProperties':False},
                    images=[str(image)],workdir=str(Path(temporary)/'smoke-test'))
                if result!={'ready':True}:raise LunaError('Synthetic image verification failed')
        try:asyncio.run(run());check('Real model image/JSON round trip',True,'ready=true')
        except Exception as exc:check('Real model image/JSON round trip',False,str(exc));return 1
    else:print('Model access has not been called. Add --smoke-test to verify authentication, image input and structured output.')
    return 0


if __name__=='__main__':raise SystemExit(main())
