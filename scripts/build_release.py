#!/usr/bin/env python3
"""Export an explicit source-only release; never copy uploads or Git history."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
TESTS=['test_agent_runtime.py','test_browser.py','test_browser_grounding.py','test_pdf_figures.py',
       'test_scatter_template_tool.py','test_scatter_templates.py','test_core.py','test_complete_plot.py','test_browser_watch.cjs',
       'test_timing.py','test_timing_view.cjs','test_scatter_beta.py']
TOP=['README.md','AGENTS.md','CLAUDE.md','pyproject.toml','uv.lock','.gitignore','plot-reader']
SCRIPTS=['setup.py','doctor.py','build_release.py','example_plot.py']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();destination=args.destination.expanduser().resolve()
    if destination.exists():raise SystemExit('Choose a new destination; existing directories are never overwritten.')
    files=[ROOT/p for p in TOP]+[ROOT/'scripts'/p for p in SCRIPTS]+[ROOT/'tests'/p for p in TESTS]
    files.append(ROOT/'experiments/scatter_templates.py')
    files.append(ROOT/'docs/scatter-beta.md')
    for path in (ROOT/'plot_reader').rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.pyc':continue
        if path.suffix not in {'.py','.json','.toml','.html','.css','.js','.md'} and path.name!='plot-reader':continue
        files.append(path)
    images=ROOT/'docs/images'
    if images.is_dir():files.extend(path for path in images.iterdir() if path.suffix=='.png')
    destination.mkdir(parents=True)
    hashes={}
    for source in sorted(set(files)):
        relative=source.relative_to(ROOT);target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);hashes[str(relative)]=hashlib.sha256(target.read_bytes()).hexdigest()
    (destination/'release-manifest.json').write_text(json.dumps({'files':hashes,'excluded':'uploads, job histories, credentials, development campaigns and source Git history'},indent=2)+'\n')
    print(json.dumps({'destination':str(destination),'files':len(hashes),'bytes':sum((destination/p).stat().st_size for p in hashes)},indent=2))


if __name__=='__main__':main()
