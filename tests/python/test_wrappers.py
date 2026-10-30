import numpy as np
import pytest
from gymnasium.wrappers import FlattenObservation
from robot_sim import RobotEnv, NormalizedJointPositionWrapper, DeltaJointPositionWrapper, RateLimitWrapper

@pytest.mark.parametrize("robot", ["so101","panda","ur5e"])
def test_normalized_endpoints_and_dict_flatten(robot):
    env=NormalizedJointPositionWrapper(RobotEnv(robot=robot));obs,_=env.reset(seed=42)
    assert set(obs)=={"joint_position","joint_velocity","end_effector_position","target_position"}
    np.testing.assert_array_equal(env.action(-np.ones(env.action_space.shape)),env.env.action_space.low)
    np.testing.assert_array_equal(env.action(np.ones(env.action_space.shape)),env.env.action_space.high)
    flat=FlattenObservation(env);obs,_=flat.reset(seed=42)
    assert flat.observation_space.contains(obs)
    obs,*_=flat.step(np.zeros(env.action_space.shape,np.float32))
    assert flat.observation_space.contains(obs)
    with pytest.raises(ValueError):env.step(np.full(env.action_space.shape,np.nan))
    with pytest.raises(ValueError):env.step(np.full(env.action_space.shape,1.01))
    env.close()

@pytest.mark.parametrize("robot", ["so101","panda","ur5e"])
def test_delta_rate_limits_composition_and_reset(robot):
    base=RobotEnv(robot=robot);rate=RateLimitWrapper(base,max_change=0.01)
    env=DeltaJointPositionWrapper(rate,max_delta=0.1)
    obs,_=env.reset(seed=42);previous=rate.previous.copy()
    requested=env.action(np.ones(env.action_space.shape))
    expected=np.clip(np.asarray(base.sim.state()["position"])+0.1,base.action_space.low,base.action_space.high)
    np.testing.assert_allclose(requested,expected,atol=2e-7)
    for _ in range(5):
        *_,info=env.step(np.ones(env.action_space.shape,np.float32))
        assert np.all(np.abs(info["joint_target"]-previous)<=0.010001)
        assert base.action_space.contains(info["joint_target"])
        previous=info["joint_target"].copy()
    env.reset(seed=42)
    np.testing.assert_allclose(rate.previous,obs["joint_position"])
    env.close()
