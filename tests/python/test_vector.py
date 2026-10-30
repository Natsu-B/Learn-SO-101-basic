import numpy as np
import pytest
from gymnasium.vector import SyncVectorEnv
from robot_sim import make_env, TaskConfig

@pytest.mark.parametrize("robot",["so101","panda","ur5e"])
def test_sync_vector_reset_step_and_autoreset(robot):
    config=TaskConfig(max_episode_steps=3)
    env=SyncVectorEnv([lambda:make_env(robot,"reach",seed=1,task_config=config),lambda:make_env(robot,"reach",seed=2,task_config=config)])
    obs,_=env.reset(seed=42)
    n=7 if robot=="panda" else 6
    assert obs["joint_position"].shape==(2,n)
    assert not np.array_equal(obs["target_position"][0],obs["target_position"][1])
    for _ in range(10):
        obs,reward,terminated,truncated,info=env.step(np.zeros((2,n),np.float32))
        assert np.isfinite(reward).all()
        assert env.observation_space.contains(obs)
    env.close()
