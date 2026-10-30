"""Record SI measurements; no real driver or learning dataset is involved."""
import argparse
import math
from pathlib import Path
from robot_sim import Simulator, SineTrajectory

p=argparse.ArgumentParser()
p.add_argument('--robot',choices=['so101','panda','ur5e'],default='so101')
p.add_argument('--joint',default='shoulder_pan')
p.add_argument('--duration',type=float,default=10)
p.add_argument('--amplitude',type=float,default=.2)
p.add_argument('--frequency',type=float,default=.5)
p.add_argument('--control-hz',type=float,default=30)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if not math.isfinite(a.control_hz) or a.control_hz<=0: p.error('control-hz must be finite and positive')
sim=Simulator(robot=a.robot)
try:
    trajectory=SineTrajectory(sim.home_position,sim.action_bounds,sim.joint_names,a.joint,a.amplitude,a.frequency,a.duration)
    # Settle at home before measuring excitation, using native position servos.
    for _ in range(60): sim.step_control(sim.home_position,1/a.control_hz)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    sim.start_trajectory(a.output,1/a.control_hz,a.seed)
    for i in range(math.ceil(a.duration*a.control_hz)):
        sim.step_control(trajectory.sample(i/a.control_hz),1/a.control_hz)
    sim.finish_trajectory()
    print(a.output)
finally: sim.close()
