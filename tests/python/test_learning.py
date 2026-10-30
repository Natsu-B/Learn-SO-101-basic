import pytest
pytest.importorskip("stable_baselines3", reason="install the optional learning extra")
from stable_baselines3 import PPO
from robot_sim import TaskConfig
from robot_sim.learning import train, evaluate


@pytest.mark.parametrize("robot", ["so101", "panda", "ur5e"])
def test_ppo_optimization_checkpoint_roundtrip(tmp_path, robot):
    result=train(robot,64,tmp_path,rollout_steps=64,batch_size=32,n_epochs=1,
                 task_config=TaskConfig(max_episode_steps=16))
    assert result["optimizer_updates"]>=1
    assert result["training_steps"]==64
    model=PPO.load(result["checkpoint"],device="cpu")
    assert model.num_timesteps==64
    assert result["episodes"]
    evaluation=evaluate(result["checkpoint"],robot=robot,episodes=2)
    assert evaluation["training_steps"]==64
    for policy in ("random", "trained"):
        assert len(evaluation[policy]["outcomes"])==2
        assert 0 <= evaluation[policy]["success_rate"] <= 1
        assert evaluation[policy]["mean_final_distance"] >= 0
    with pytest.raises(ValueError,match="robot"):
        evaluate(result["checkpoint"],robot="wrong",episodes=1)


def test_optional_wandb_metrics_without_login_or_network(tmp_path,monkeypatch):
    import wandb
    logs=[]
    class Run:
        def log(self,data,step): logs.append(data)
        def finish(self): pass
    monkeypatch.setattr(wandb,"init",lambda **kw: Run())
    train("so101",32,tmp_path,rollout_steps=32,batch_size=16,n_epochs=1,
          task_config=TaskConfig(max_episode_steps=8),wandb_enabled=True)
    assert logs
    assert {"episode/reward","episode/length","episode/final_distance","rollout/success_rate","train/learning_rate"} <= logs[0].keys()

@pytest.mark.parametrize('robot',['so101','panda','ur5e'])
def test_domain_randomized_ppo_and_robustness(tmp_path,robot):
    from robot_sim.randomization import load_profile
    result=train(robot,64,tmp_path,rollout_steps=64,batch_size=32,n_epochs=1,
                 task_config=TaskConfig(max_episode_steps=16),domain_randomization='moderate')
    for profile in ('none','light','moderate'):
        report=evaluate(result['checkpoint'],robot=robot,episodes=2,domain_randomization=profile)
        assert report['domain_randomization']==load_profile(profile)
        assert report['trained']['mean_final_distance']>=0


def test_sb3_sac_smoke():
    from stable_baselines3 import SAC
    from robot_sim import make_env
    import torch
    torch.set_num_threads(1)
    env=make_env('so101',domain_randomization='light',task_config=TaskConfig(max_episode_steps=8))
    try:
        model=SAC('MultiInputPolicy',env,learning_starts=8,buffer_size=128,batch_size=8,
                  policy_kwargs={'net_arch':[16,16]},device='cpu',seed=42)
        model.learn(32)
        assert model._n_updates>0
    finally: env.close()
