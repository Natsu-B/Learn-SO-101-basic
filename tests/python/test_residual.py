import numpy as np
import pytest
from robot_sim import RobotEnv,ResidualActionWrapper

@pytest.mark.parametrize('robot',['so101','panda','ur5e'])
def test_residual_zero_limits_range_rate(robot):
    base=RobotEnv(robot=robot);target=np.asarray(base.sim.state()['position'],np.float32)
    env=ResidualActionWrapper(base,lambda obs:target,max_residual=.01,max_change=.005)
    env.reset(seed=42)
    result=env.step(np.zeros(env.action_space.shape))
    np.testing.assert_allclose(result[4]['joint_target'],target,atol=2e-7)
    previous=result[4]['joint_target'].copy()
    for _ in range(3):
        result=env.step(np.ones(env.action_space.shape))
        assert np.max(abs(result[4]['residual_joint_position']))<=.01
        assert np.max(abs(result[4]['joint_target']-previous))<=.005001
        assert base.action_space.contains(result[4]['joint_target'])
        previous=result[4]['joint_target'].copy()
    target[:]=1e6
    result=env.step(np.ones(env.action_space.shape))
    assert base.action_space.contains(result[4]['joint_target'])
    assert np.max(abs(result[4]['joint_target']-previous))<=.005001
    with pytest.raises(ValueError): env.step(np.full(env.action_space.shape,np.nan))
    env.close()
