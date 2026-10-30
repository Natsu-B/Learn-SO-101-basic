import json
import numpy as np
import pytest
from robot_sim import RobotEnv,make_env
from robot_sim.randomization import load_profile

@pytest.mark.parametrize('robot',['so101','panda','ur5e'])
def test_dr_reproducible_and_hidden_from_observation(robot):
    env=make_env(robot,domain_randomization='moderate')
    obs,info=env.reset(seed=42);p=info['domain_randomization']
    assert set(obs)=={'joint_position','joint_velocity','end_effector_position','target_position'}
    assert env.reset(seed=42)[1]['domain_randomization']==p
    assert env.reset(seed=43)[1]['domain_randomization']!=p
    assert .9<=p['actuator_gain_scale'][0]<=1.1
    for _ in range(100):
        obs,_,term,trunc,info=env.step(np.zeros(env.action_space.shape))
        assert env.observation_space.contains(obs)
        assert info['observation_time']<=info['time']
        if term or trunc: env.reset()
    env.close()

def test_parameter_file_center_and_invalid_config(tmp_path):
    env=RobotEnv();env.reset(seed=42);path=tmp_path/'params.json';env.sim.save_parameters(path);env.close()
    p=json.loads(path.read_text());p['joint_damping_scale']=[1.15]*6;path.write_text(json.dumps(p))
    env=RobotEnv(model_parameters=path,domain_randomization={'joint_damping_scale':{'min':.9,'max':.9}})
    assert env.reset(seed=42)[1]['domain_randomization']['joint_damping_scale'][0]==pytest.approx(1.15*.9)
    env.close()
    with pytest.raises(ValueError): RobotEnv(domain_randomization={'unknown':{'min':0,'max':1}})
    with pytest.raises(ValueError): RobotEnv(domain_randomization={'actuator_gain_scale':{'min':2,'max':1}})


def test_bare_reset_applies_identified_center():
    from robot_sim import Simulator
    sim=Simulator()
    p=json.loads(sim.parameters_json());p['control_latency_seconds']=.02
    sim.set_parameter_center(json.dumps(p));sim.reset()
    assert json.loads(sim.parameters_json())==p
    sim.close()
