"""Same held-out target seeds, fixed center vs light/moderate perturbations."""
import argparse
import json
from pathlib import Path
from robot_sim.learning import evaluate
p=argparse.ArgumentParser()
p.add_argument('--robot',choices=['so101','panda','ur5e'])
p.add_argument('--model',required=True)
p.add_argument('--episodes',type=int,default=100)
p.add_argument('--seed',type=int,default=20000)
p.add_argument('--output',type=Path)
a=p.parse_args()
directory=Path(a.model) if Path(a.model).is_dir() else Path(a.model).parent
config=json.loads((directory/'config.json').read_text())
if config.get('goal_environment'):
    from robot_sim.precision import evaluate_goal as evaluate
results={profile:evaluate(a.model,robot=a.robot,episodes=a.episodes,seed=a.seed,domain_randomization=profile)
         for profile in ('none','light','moderate')}
if a.output:
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
for profile,result in results.items():
    print(profile,json.dumps({k:v for k,v in result['trained'].items() if k!='outcomes'}))
