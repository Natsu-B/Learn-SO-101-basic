"""Fixed-seed precision evaluation. No target selection after seeing outcomes."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from .goal_env import GoalReachEnv
from .config import TaskConfig


class HoldTracker:
    """Consecutive in-range control samples, timed by MuJoCo simulation time.

    This checks sampled positions, not an unsampled physics trajectory. Leaving
    the threshold resets the timer; the policy keeps acting during the hold.
    """
    def __init__(self, threshold, seconds):
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError('threshold must be finite and positive')
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError('hold seconds must be finite and non-negative')
        self.threshold = threshold
        self.seconds = seconds
        self.started = None
        self.first_reach_time = None
        self.best_duration = 0.
        self.errors = []
        self.completed = False

    def update(self, time, distance):
        if distance <= self.threshold:
            if self.first_reach_time is None:
                self.first_reach_time = time
            if self.started is None:
                self.started = time
            self.best_duration = max(self.best_duration, time-self.started)
            self.completed = time-self.started + 1e-12 >= self.seconds
        else:
            self.started = None
            self.completed = False
        if self.first_reach_time is not None:
            self.errors.append(float(distance))
        return self.completed

    def metrics(self):
        return dict(reached=self.first_reach_time is not None,
                    first_reach_time=self.first_reach_time,
                    hold_success=self.completed,
                    longest_hold_seconds=self.best_duration,
                    post_reach_mean_distance=float(np.mean(self.errors)) if self.errors else None,
                    post_reach_max_distance=max(self.errors) if self.errors else None)


def success_at_distance(distance, threshold=.005):
    return np.asarray(distance) <= threshold


def summarize(outcomes, threshold=.005):
    if not outcomes:
        raise ValueError('evaluation requires at least one episode')
    d=np.asarray([x['final_distance'] for x in outcomes])
    success=np.asarray([x.get('success', bool(x['final_distance'] <= threshold)) for x in outcomes])
    result=dict(success_rate=float(success.mean()),
                mean_episode_reward=float(np.mean([x['reward'] for x in outcomes])),
                mean_episode_length=float(np.mean([x['length'] for x in outcomes])),
                mean_final_distance=float(d.mean()),
                median_final_distance=float(np.median(d)),
                max_final_distance=float(d.max()),
                percentiles_m={f'P{p}':float(np.percentile(d,p)) for p in (50,90,95,99)},
                within_m={str(t):float(np.mean(d<=t)) for t in (.002,.005,.01,.02)},
                distance_bins={},outcomes=outcomes)
    if 'hold_success' in outcomes[0]:
        result['reach_success_rate']=float(np.mean([x['reached'] for x in outcomes]))
        result['hold_success_rate']=float(np.mean([x['hold_success'] for x in outcomes]))
    initial=np.asarray([x['initial_distance'] for x in outcomes])
    for lo,hi in ((.05,.1),(.1,.15),(.15,.2),(.2,.25)):
        mask=(initial>=lo)&((initial<hi) if hi<.25 else (initial<=hi))
        result['distance_bins'][f'{lo:.2f}-{hi:.2f}']={'count':int(mask.sum()),'success_rate':float(success[mask].mean()) if mask.any() else None}
    return result


def goal_environment(config, *, evaluation=False, stage=None, threshold=None):
    task=TaskConfig(**config['task_config'])
    if stage is not None:
        task=replace(task,success_threshold=stage['threshold'],min_target_distance=stage['minimum'],max_target_distance=stage['maximum'])
    if threshold is not None:
        task=replace(task,success_threshold=threshold,min_target_distance=.05,max_target_distance=.25)
    return GoalReachEnv(config['robot'],max_delta=config['max_delta'],distance_scale=config['distance_scale'],
                        rate_limit=config['rate_limit'],control_frequency=config['control_frequency'],
                        terminate_on_success=evaluation,task_config=task,domain_randomization=config.get('domain_randomization'))


def rollout(model,config,*,episodes=64,seed=1000000,stage=None,threshold=None,random=False,hold_seconds=0.):
    HoldTracker(threshold if threshold is not None else config['task_config']['success_threshold'], hold_seconds)
    env=goal_environment(config,evaluation=hold_seconds == 0,stage=stage,threshold=threshold)
    outcomes=[]
    try:
        for i in range(episodes):
            obs,info=env.reset(seed=seed+i);env.action_space.seed(seed+i)
            initial=info['distance'];total=0.
            hold=HoldTracker(env.task_config.success_threshold, hold_seconds)
            for length in range(1,env.task_config.max_episode_steps+1):
                action=env.action_space.sample() if random else model.predict(obs,deterministic=True)[0]
                obs,r,done,truncated,info=env.step(action);total+=r
                held=hold.update(env.last_result['time'], info['distance'])
                if held or done or truncated:break
            outcomes.append(dict(seed=seed+i,initial_distance=initial,reward=total,length=length,final_distance=info['distance'],success=held, **hold.metrics()))
        return summarize(outcomes,env.task_config.success_threshold)
    finally:env.close()


def log_selected(model,config,outcomes,directory,threshold=.005,spawn=False,hold_seconds=0.):
    """Replay the best/worst seeds deterministically. Rust logs true/applied state."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    ordered=sorted(outcomes,key=lambda x:x['final_distance'])
    selected=[('best',x) for x in ordered[:5]]+[('worst',x) for x in ordered[-5:]]
    files=[]
    for category,x in selected:
        env=goal_environment(config,evaluation=hold_seconds == 0,threshold=threshold)
        hold=HoldTracker(threshold, hold_seconds)
        path=directory/f'{category}-{x["seed"]}.rrd'
        try:
            env.sim.start_rerun(record=path)
            obs,_=env.reset(seed=x['seed'])
            for _ in range(env.task_config.max_episode_steps):
                obs,_,done,truncated,info=env.step(model.predict(obs,deterministic=True)[0])
                if hold.update(env.last_result['time'],info['distance']) or done or truncated:break
            env.sim.flush_telemetry();files.append(str(path))
        finally:env.close()
    if spawn and files:
        import subprocess
        subprocess.Popen(['rerun',files[-1]])
    return files


def evaluate_goal(checkpoint,*,episodes=500,seed=2000000,threshold=.005,robot=None,rerun=False,record=None,domain_randomization=None,hold_seconds=0.):
    HoldTracker(threshold, hold_seconds)
    if episodes <= 0:
        raise ValueError('episodes must be positive')
    import torch
    from stable_baselines3 import SAC,PPO
    checkpoint=Path(checkpoint)
    if checkpoint.is_dir():checkpoint=checkpoint/'best.zip'
    config=json.loads((checkpoint.parent/'config.json').read_text())
    if robot is not None and robot!=config['robot']:raise ValueError('robot does not match checkpoint')
    if domain_randomization is not None:config['domain_randomization']=domain_randomization
    torch.set_num_threads(1)
    load_env=goal_environment(config,evaluation=True,threshold=threshold)
    try:
        model=(PPO if config['algorithm']=='ppo' else SAC).load(checkpoint,env=load_env,device='cpu')
    finally:
        # Prediction does not execute model.env; rollouts own fresh environments.
        load_env.close()
    result=dict(robot=config['robot'],algorithm=config['algorithm'],training_steps=model.num_timesteps,episodes=episodes,first_seed=seed,threshold_m=threshold,hold_seconds=hold_seconds,hold_sampling="control-step endpoints",domain_randomization=config.get('domain_randomization'),
                random=rollout(model,config,episodes=episodes,seed=seed,threshold=threshold,random=True,hold_seconds=hold_seconds),
                trained=rollout(model,config,episodes=episodes,seed=seed,threshold=threshold,hold_seconds=hold_seconds))
    if rerun or record is not None:
        result['rerun_files']=log_selected(model,config,result['trained']['outcomes'],record or checkpoint.parent/'evaluation-rerun',threshold,spawn=rerun,hold_seconds=hold_seconds)
    return result
