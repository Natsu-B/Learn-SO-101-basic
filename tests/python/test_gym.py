import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from robot_sim import RobotEnv, Simulator

@pytest.mark.parametrize("robot,n", [("so101",6),("panda",7),("ur5e",6)])
def test_gym_smoke_spaces_and_long_stepping(robot,n):
    env = RobotEnv(robot=robot)
    assert env.action_space.shape == (n,)
    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs)
    env.action_space.seed(42)
    for _ in range(1000):
        action = env.action_space.sample()
        assert env.action_space.contains(action)
        obs,reward,terminated,truncated,info = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        assert all(np.isfinite(v).all() for v in obs.values())
        assert reward == pytest.approx(-np.linalg.norm(env.sim.end_effector-np.asarray(env.sim.target)))
        if terminated or truncated: env.reset()
    env.close()

@pytest.mark.parametrize("robot", ["so101","panda","ur5e"])
def test_gym_checker_and_seed(robot):
    env=RobotEnv(robot=robot)
    check_env(env, skip_render_check=True)
    first, _ = env.reset(seed=42)
    command=first["joint_position"]
    sequence=[env.step(command)[0] for _ in range(10)]
    second,_=env.reset(seed=42)
    for key in first: np.testing.assert_array_equal(first[key], second[key])
    for expected in sequence:
        actual=env.step(command)[0]
        for key in actual: np.testing.assert_array_equal(actual[key],expected[key])
    third,_=env.reset(seed=43)
    assert not np.array_equal(first["target_position"],third["target_position"])
    env.close()

@pytest.mark.parametrize("bad", [[0]*5,[np.nan]*6,[np.inf]*6,[100]*6,["bad"]*6,[1j]*6])
def test_invalid_action_preserves_time(bad):
    env=RobotEnv();env.reset(seed=42)
    with pytest.raises(ValueError): env.step(bad)
    assert env.sim.time==0
    env.close()

def test_control_period_and_truncation():
    env=RobotEnv(max_episode_steps=3)
    obs,_=env.reset(seed=42)
    for _ in range(3): result=env.step(obs["joint_position"])
    assert result[3]
    assert env.sim.time==pytest.approx(0.1)
    env.close()

@pytest.mark.parametrize("robot", ["so101","panda","ur5e"])
def test_native_dimension_and_reach(robot):
    sim=Simulator(robot=robot)
    sim.reset_reach(42)
    initial=sim.state(); target=sim.target
    sim.step_control(initial["position"],1/30)
    assert sim.time>0
    sim.reset_reach(42)
    assert sim.state()==initial and sim.target==target
    with pytest.raises(ValueError): sim.step([0])


def test_explicit_seed_matches_native_rust_task_and_none_continues_rng():
    env = RobotEnv()
    first, _ = env.reset(seed=42)
    sim = Simulator()
    sim.reset_reach(42)
    np.testing.assert_array_equal(first["target_position"], np.asarray(sim.target, np.float32))
    second, _ = env.reset()
    assert not np.array_equal(first["target_position"], second["target_position"])
    env.close()


def test_task_config_success_bonus_and_horizon():
    from robot_sim import TaskConfig
    env=RobotEnv(task_config=TaskConfig(success_threshold=1.0, success_bonus=2.0,max_episode_steps=1))
    obs,_=env.reset(seed=42)
    _,reward,terminated,truncated,info=env.step(obs["joint_position"])
    assert terminated and not truncated
    assert reward==pytest.approx(2.0-info["distance"])
    env.close()
    env=RobotEnv(task_config=TaskConfig(max_episode_steps=1))
    obs,_=env.reset(seed=42)
    _,_,terminated,truncated,_=env.step(obs["joint_position"])
    assert not terminated and truncated
    env.close()
