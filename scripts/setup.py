#!/usr/bin/env python3
"""Install project dependencies with uv, check the chosen agent, optionally run."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent',choices=['codex','claude'],required=True)
    parser.add_argument('--smoke-test',action='store_true')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    uv=shutil.which('uv')
    if not uv:raise SystemExit('Install uv from https://docs.astral.sh/uv/getting-started/installation/ and rerun this command.')
    subprocess.run([uv,'sync','--extra','test'],cwd=ROOT,check=True)
    check=[uv,'run','python','scripts/doctor.py','--agent',args.agent]
    if args.smoke_test:check.append('--smoke-test')
    subprocess.run(check,cwd=ROOT,check=True)
    if args.run:
        os.chdir(ROOT);os.execv(uv,[uv,'run','plot-reader','web','--agent',args.agent])
    print(f'Ready. Start with: uv run plot-reader web --agent {args.agent}')


if __name__=='__main__':main()
