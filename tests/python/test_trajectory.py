import numpy as np
import pytest
from robot_sim.trajectory import Trajectory,compare,log_comparison
from robot_sim import Simulator
from robot_sim._native import SineTrajectory

def trace(offset=0,delay=0):
    t=np.arange(0,3,.01);q=np.sin(2*np.pi*.7*(t-delay))+offset
    m=dict(schema_version=1,robot='fixture',backend='real',model_version='synthetic',joint_names=['j'],control_period=.01,physics_timestep=None,timestamp=1,seed=42,randomization_parameters={})
    return Trajectory(m,np.column_stack([t,q,q,2*np.pi*.7*np.cos(2*np.pi*.7*(t-delay)),np.full((len(t),3),np.nan)]))

def test_comparison_alignment_and_known_errors(tmp_path):
    a=trace();a.save(tmp_path/'a');b=Trajectory.from_csv(tmp_path/'a')
    assert b.metadata==a.metadata
    assert len(b.time)==300
    assert compare(a,b)[0]['j']['rmse_position']==0
    assert compare(a,trace(offset=.2))[0]['j']['rmse_position']==pytest.approx(.2)
    assert compare(a,trace(delay=.08))[0]['j']['estimated_time_lag']==pytest.approx(.08,abs=.011)
    with pytest.raises(ValueError): Trajectory(a.metadata,a.samples[::-1])
    with pytest.raises(ValueError): Trajectory(a.metadata,a.samples[:,:-1])

def test_rust_record_python_reload_and_safe_excitation(tmp_path):
    sim=Simulator(robot='panda');home=sim.home_position
    wave=SineTrajectory(home,sim.action_bounds,sim.joint_names,'joint1',.1,.5,1)
    with pytest.raises(ValueError): SineTrajectory(home,sim.action_bounds,sim.joint_names,'joint1',100,.5,1)
    sim.start_trajectory(tmp_path/'sim',.01,42)
    for i in range(100): sim.step_control(wave.sample(i*.01),.01)
    sim.finish_trajectory();sim.close()
    a=Trajectory.from_csv(tmp_path/'sim')
    assert len(a.time)==100 and a.metadata['backend']=='mujoco'
    assert a.position.shape==(100,7)
    assert compare(a,a)[0]['joint1']['rmse_position']==0

def test_comparison_rerun_headless(tmp_path):
    p=tmp_path/'comparison.rrd';log_comparison(trace(),trace(offset=.1),record=p)
    assert p.stat().st_size>100

def test_v2_command_acquisition_and_delivery_are_distinct(tmp_path):
    import json
    from robot_sim.trajectory import zero_order_hold
    sim=Simulator()
    parameters=json.loads(sim.parameters_json())
    parameters['control_latency_seconds']=.04
    parameters['observation_latency_seconds']=.03
    sim.set_parameter_center(json.dumps(parameters));sim.reset()
    sim.start_trajectory(tmp_path/'v2',.01,42)
    command=sim.home_position;command[0]+=.2
    for _ in range(20):sim.step_control(command,.01)
    sim.finish_trajectory();sim.close()
    a=Trajectory.from_csv(tmp_path/'v2')
    assert a.metadata['schema_version']==2
    assert np.all(a.command_time<=a.time)
    assert np.all(a.applied_time<=a.time)
    assert np.all(a.samples[:,3]<=a.time)
    assert np.all(np.diff(a.time)>0)
    assert np.any(a.command[:,0]!=a.applied[:,0])
    assert np.any(a.samples[:,3]<a.time)
    a.save(tmp_path/'copy');b=Trajectory.from_csv(tmp_path/'copy')
    assert a.metadata==b.metadata
    assert np.allclose(a.samples,b.samples,equal_nan=True)
    # A command jump is held until its timestamp, never interpolated.
    assert np.array_equal(zero_order_hold(np.array([.49,.5,.9]),np.array([0.,.5]),np.array([0.,1.])),[0,1,1])


def test_sensor_offset_alias_reads_but_writes_unambiguous_name():
    import json
    sim=Simulator()
    try:
        sim.configure_randomization(json.dumps({'joint_zero_offset_rad':{'min':.01,'max':.01}}))
        sim.reset()
        p=json.loads(sim.parameters_json())
        assert p['sensor_zero_offset_rad']==[.01]*6
        assert 'joint_zero_offset_rad' not in p
    finally:sim.close()
