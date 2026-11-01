#!/usr/bin/env python3
"""Generate a synthetic scatter plot without any private source material."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path('artifacts/example-scatter.png'))
args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
figure,axis=plt.subplots(figsize=(6,4),dpi=140)
axis.scatter([1,3,6,8],[1,2,4,7],s=60,facecolors='white',edgecolors='black',linewidths=1.6,label='Sample A')
axis.set(xlim=(0,10),ylim=(0,10),xlabel='Time (s)',ylabel='Signal (a.u.)')
axis.set_xticks([0,2,4,6,8,10]);axis.set_yticks([0,2,4,6,8,10]);axis.legend(loc='upper left')
figure.tight_layout();figure.savefig(args.output);plt.close(figure)
print(args.output.resolve())
