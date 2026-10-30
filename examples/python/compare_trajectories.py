import argparse
import json
from robot_sim.trajectory import Trajectory,compare,log_comparison
p=argparse.ArgumentParser()
p.add_argument('--sim',required=True);p.add_argument('--real',required=True)
p.add_argument('--max-lag',type=float,default=.2)
p.add_argument('--absolute-timestamps',action='store_true')
p.add_argument('--rerun',action='store_true');p.add_argument('--record')
a=p.parse_args()
sim=Trajectory.from_csv(a.sim);real=Trajectory.from_csv(a.real)
kw=dict(max_lag=a.max_lag,absolute_timestamps=a.absolute_timestamps)
result=log_comparison(sim,real,record=a.record,**kw) if a.rerun or a.record else compare(sim,real,**kw)[0]
print(json.dumps(result,indent=2,allow_nan=False))
