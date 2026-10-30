import argparse,json,time
from pathlib import Path
import numpy as np
from robot_sim import Simulator
from robot_sim.ik import DlsController
from robot_sim.selection import reach_actuators
p=argparse.ArgumentParser();p.add_argument('--robot',default='so101');p.add_argument('--episodes',type=int,default=500)
p.add_argument('--threshold',type=float,default=.005);p.add_argument('--seed',type=int,default=1000000)
p.add_argument('--episode-steps',type=int,default=300);p.add_argument('--output',type=Path)
a=p.parse_args();sim=Simulator(robot=a.robot);names=sim.actuator_names;selected=reach_actuators(a.robot,names)
sim.configure_reach(success_threshold=a.threshold,max_episode_steps=a.episode_steps)
sim.configure_reach_selection(selected);indices=[names.index(n) for n in selected];out=[];start=time.monotonic()
try:
 for i in range(a.episodes):
    sim.reset_reach(a.seed+i);controller=DlsController(sim,indices)
    for length in range(1,a.episode_steps+1):
        r=sim.step_task(controller.action(sim.target),1/30)
        if r['distance']<=a.threshold:break
    out.append(dict(seed=a.seed+i,distance=r['distance'],success=r['distance']<=a.threshold,length=length))
    if (i+1)%50==0:print('IK',i+1,sum(x['success'] for x in out)/len(out),flush=True)
finally:sim.close()
d=np.array([x['distance'] for x in out]);result=dict(robot=a.robot,episodes=a.episodes,threshold=a.threshold,success_rate=float(np.mean([x['success'] for x in out])),mean_final_distance=float(d.mean()),median_final_distance=float(np.median(d)),p95_final_distance=float(np.percentile(d,95)),seconds=time.monotonic()-start,outcomes=out)
if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print({k:v for k,v in result.items() if k!='outcomes'})
