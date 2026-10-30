import numpy as np
import pytest
from robot_sim import GoalReachEnv, TaskConfig
from robot_sim.ik import DlsController

@pytest.mark.parametrize('robot,dim',[('so101',5),('panda',7),('ur5e',6)])
def test_goal_contract(robot,dim):
    env=GoalReachEnv(robot,task_config=TaskConfig(success_threshold=.005,max_episode_steps=3))
    try:
        obs,info=env.reset(seed=42)
        assert set(obs)=={'observation','achieved_goal','desired_goal'}
        assert obs['achieved_goal'].shape==obs['desired_goal'].shape==(3,)
        assert obs['observation'].shape==(dim*2+3,)
        assert env.action_space.shape==(dim,)
        assert env.observation_space.contains(obs)
        target=obs['desired_goal'].copy()
        for _ in range(3):
            obs,r,terminated,truncated,info=env.step(np.zeros(dim,np.float32))
            assert r==env.compute_reward(obs['achieved_goal'],obs['desired_goal'],info)
            assert np.array_equal(obs['desired_goal'],target)
            assert not terminated
        assert truncated
        if robot=='so101':
            # Moving gripper's actual servo error is not a policy command.
            assert env.previous_command[5]==env.home[5]
        a,_=env.reset(seed=42);b,_=env.reset(seed=42)
        assert all(np.array_equal(a[k],b[k]) for k in a)
        with pytest.raises(ValueError):env.step(np.full(dim,np.nan))
    finally:env.close()

def test_goal_vector_reward_boundary_and_termination():
    e=GoalReachEnv(terminate_on_success=True,task_config=TaskConfig(success_threshold=.005,success_bonus=1))
    try:
        g=np.zeros((3,3));a=np.array([[.004,0,0],[.005,0,0],[.006,0,0]])
        assert np.allclose(e.compute_reward(a,g,[{}]*3),[.96,.95,-.06])
        assert np.array_equal(e.compute_terminated(a,g,[{}]*3),[True,True,False])
        assert np.array_equal(e.compute_truncated(a,g,[{'episode_step':0},{'episode_step':1000},{'episode_step':0}]),[False,True,False])
    finally:e.close()

def test_dls_reaches_without_hidden_target_configuration():
    e=GoalReachEnv(terminate_on_success=True)
    try:
        obs,_=e.reset(seed=1000000)
        c=DlsController(e.sim,e.indices)
        for _ in range(300):
            r=e.sim.step_task(c.action(obs['desired_goal']),1/30)
            if r['distance']<=.005:break
        assert r['distance']<=.005
        assert 'target_configuration' not in r
    finally:e.close()

def test_evaluation_termination_matches_goal_semantics():
    env=GoalReachEnv(terminate_on_success=True)
    try:
        obs,_=env.reset(seed=1000000);controller=DlsController(env.sim,env.indices)
        for _ in range(100):
            q=np.asarray(env.last_result['joint_position'])
            command=np.asarray(controller.action(obs['desired_goal']))
            action=np.clip((command[env.indices]-q[env.indices])/env.max_delta,-1,1).astype(np.float32)
            obs,_,terminated,truncated,info=env.step(action)
            assert terminated==bool(env.compute_terminated(obs['achieved_goal'],obs['desired_goal'],info))
            if terminated or truncated:break
        assert terminated and info['distance']<=.005
    finally:env.close()

@pytest.mark.parametrize('robot',['so101','panda','ur5e'])
def test_goal_sync_vector(robot):
    import gymnasium as gym
    env=gym.vector.SyncVectorEnv([lambda:GoalReachEnv(robot),lambda:GoalReachEnv(robot)])
    try:
        obs,_=env.reset(seed=42)
        assert env.observation_space.contains(obs)
        for _ in range(10):
            obs,_,_,_,_=env.step(np.zeros(env.action_space.shape,np.float32))
            assert env.observation_space.contains(obs)
    finally:env.close()
