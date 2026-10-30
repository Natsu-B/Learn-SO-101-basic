"""Optional static evaluation plot: uv sync --extra learning --extra analysis --no-install-project."""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--evaluation',type=Path,default=Path('docs/results/reach5mm-final.json'))
p.add_argument('--output',type=Path,default=Path('docs/results/reach5mm-error-cdf'))
a=p.parse_args()
r=json.loads(a.evaluation.read_text())
fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
for policy,color in [('trained','#1565c0'),('random','#999999')]:
 d=np.sort([o['final_distance']*1000 for o in r[policy]['outcomes']])
 ax.step(d,np.arange(1,len(d)+1)/len(d)*100,where='post',label=policy,color=color,lw=2)
ax.axvline(5,color='#b71c1c',linestyle='--',label='5 mm criterion')
ax.axhline(90,color='#555555',linestyle=':',lw=1)
ax.set(xlim=(0,20),ylim=(0,101),xlabel='Final end-effector error (mm)',ylabel='Episodes within error (%)',title='SO-101 Reach: 500 fixed held-out episodes')
ax.grid(alpha=.2);ax.legend(loc='lower right')
success=sum(x['final_distance']<=r['threshold_m'] for x in r['trained']['outcomes'])
ax.text(8,92,f'{r["algorithm"]}: {success} / {r["episodes"]} ({100*success/r["episodes"]:.1f}%)',color='#1565c0')
a.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(a.output.with_suffix('.png'),dpi=180)
fig.savefig(a.output.with_suffix('.pdf'))
