import json
import pytest
pytest.importorskip('stable_baselines3')
from stable_baselines3 import SAC,HerReplayBuffer
from robot_sim.goal_learning import train_goal
from robot_sim.precision import goal_environment,evaluate_goal,summarize,success_at_distance


@pytest.mark.parametrize('robot',['so101','panda','ur5e'])
def test_her_checkpoint_replay_resume(tmp_path,robot):
    args=dict(robot=robot,steps=64,max_steps=128,horizon=8,eval_interval=64,eval_episodes=2,buffer_size=256,batch_size=8,
              learning_starts=8,net_arch=[16,16],train_freq=4,gradient_steps=1,final_eval_episodes=2)
    state=train_goal(tmp_path,**args)
    assert state['training_steps']==64
    config=json.loads((tmp_path/'config.json').read_text());env=goal_environment(config)
    try:
        model=SAC.load(tmp_path/'latest.zip',env=env,device='cpu')
        model.load_replay_buffer(tmp_path/'latest-replay.pkl')
        assert isinstance(model.replay_buffer,HerReplayBuffer)
        assert model._n_updates>0
        assert model.replay_buffer.size()==64
        assert model.replay_buffer.sample(8).rewards.shape==(8,1)
    finally:env.close()
    resumed=train_goal(tmp_path,resume=tmp_path/'latest.zip',**args)
    assert resumed['training_steps']==128
    assert resumed['target_rng_state']!=state['target_rng_state']
    assert 'numpy_rng_state' in resumed and 'torch_rng_state' in resumed
    report=evaluate_goal(tmp_path/'latest.zip',episodes=2)
    assert report['training_steps']==128
    assert len(report['trained']['outcomes'])==2


def test_precision_boundary_percentiles_bins():
    assert success_at_distance(.005)
    assert not success_at_distance(.005000001)
    out=[dict(final_distance=d,initial_distance=i,reward=-d,length=10) for d,i in [(0,.06),(.005,.11),(.01,.16),(.02,.21)]]
    s=summarize(out)
    assert s['success_rate']==.5
    assert s['percentiles_m']['P50']==pytest.approx(.0075)
    assert s['percentiles_m']['P95']==pytest.approx(.0185)
    assert [x['count'] for x in s['distance_bins'].values()]==[1]*4


def test_goal_warm_start_light_uses_fresh_replay(tmp_path):
    args=dict(steps=32,max_steps=64,horizon=8,eval_interval=32,eval_episodes=2,buffer_size=128,batch_size=8,learning_starts=8,net_arch=[16,16],train_freq=4,gradient_steps=1,final_eval_episodes=2)
    train_goal(tmp_path/'nominal',**args)
    state=train_goal(tmp_path/'light',initialize_from=tmp_path/'nominal'/'best.zip',domain_randomization='light',**args)
    assert state['training_steps']==32
    config=json.loads((tmp_path/'light'/'config.json').read_text())
    assert config['pretrained_steps']==32
    assert config['domain_randomization']=='light'
    assert config['net_arch']==[16,16]

def test_artifact_explicit_units_and_action_contract(tmp_path):
    train_goal(tmp_path,steps=16,horizon=8,eval_interval=16,eval_episodes=1,final_eval_episodes=1,
               buffer_size=64,batch_size=8,learning_starts=8,net_arch=[16,16],train_freq=4,gradient_steps=1)
    artifact=json.loads((tmp_path/'artifact.json').read_text())
    assert artifact['task']=='reach'
    assert artifact['success_threshold_m']==.005
    assert artifact['physics_timestep_s']==.005
    assert artifact['control_period_s']==pytest.approx(1/30)
    assert artifact['action_mode']=='normalized_joint_delta'
    assert len(artifact['action_schema']['controlled_names'])==5
    assert artifact['internal_units']['joint_position']=='radian'


def test_hold_finetune_applies_requested_schedule_and_reward(tmp_path):
    args=dict(steps=32,max_steps=64,horizon=8,eval_interval=32,eval_episodes=2,
              buffer_size=128,batch_size=8,learning_starts=8,net_arch=[16,16],
              train_freq=4,gradient_steps=1,final_eval_episodes=2)
    train_goal(tmp_path/'source',**args)
    train_goal(tmp_path/'hold',initialize_from=tmp_path/'source'/'best.zip',
               learning_rate=.0001,target_threshold=.003,validation_threshold=.005,
               hold_seconds=.5,success_bonus=.1,**args)
    config=json.loads((tmp_path/'hold'/'config.json').read_text())
    env=goal_environment(config)
    try:
        model=SAC.load(tmp_path/'hold'/'latest.zip',env=env)
        assert model.lr_schedule(1.)==pytest.approx(.0001)
        assert model.learning_rate==.0001
        assert env.task_config.success_threshold==.003
        assert env.task_config.success_bonus==.1
        assert env.compute_reward([0,0,0],[0,0,0],{})==pytest.approx(.1)
        state=json.loads((tmp_path/'hold'/'training_state.json').read_text())
        assert state['history'][0]['hold_seconds']==.5
        assert state['history'][0]['success_rate']==0  # 8 steps cannot hold .5s.
    finally:env.close()
