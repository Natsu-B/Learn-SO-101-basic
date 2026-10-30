"""Optional SB3 SAC/HER runner with bounded, resumable curriculum training."""
from dataclasses import asdict
from pathlib import Path
import json
import time
import numpy as np
from .config import TaskConfig
from .precision import goal_environment, rollout

STAGES=[dict(name='A',minimum=.02,maximum=.08,threshold=.02),
        dict(name='B',minimum=.03,maximum=.12,threshold=.01),
        dict(name='C',minimum=.05,maximum=.20,threshold=.0075),
        dict(name='D',minimum=.05,maximum=.25,threshold=.005)]


def train_goal(output,*,robot='so101',steps=500000,max_steps=5000000,resume=None,
               initialize_from=None,keep_replay_between_stages=False,
               curriculum=False,algorithm='sac-her',max_delta=.05,seed=42,
               horizon=100,eval_interval=25000,eval_episodes=64,
               buffer_size=1000000,batch_size=256,learning_rate=.001,gamma=.98,tau=.005,
               train_freq=16,gradient_steps=4,learning_starts=1000,net_arch=(256,256,256),
               wandb_enabled=False,wandb_mode='offline',target_threshold=.005,domain_randomization=None,final_eval_episodes=500,validation_success=.9,hold_seconds=0.,success_bonus=0.,validation_threshold=None):
    import torch
    import stable_baselines3
    from stable_baselines3 import SAC,PPO,HerReplayBuffer
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    if resume is not None and initialize_from is not None:raise ValueError('resume and initialize_from are exclusive')
    if algorithm not in ('ppo','sac','sac-her'):raise ValueError('unknown algorithm')
    if not .9<=validation_success<=1:raise ValueError('validation_success must be in [0.9,1]')
    if steps<1 or max_steps<1 or max_steps>5000000 or eval_interval<1 or eval_episodes<1:raise ValueError('invalid bounded training budget')
    if not np.isfinite(target_threshold) or target_threshold <= 0:raise ValueError('target threshold must be positive')
    from .precision import HoldTracker
    validation_threshold=target_threshold if validation_threshold is None else validation_threshold
    HoldTracker(validation_threshold,hold_seconds)
    if not np.isfinite(success_bonus) or success_bonus < 0:raise ValueError('success bonus must be non-negative')
    torch.set_num_threads(1)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    state_path=output/'training_state.json'
    if resume is None and (output/'latest.zip').exists():raise FileExistsError('use --resume to continue existing training')
    config=dict(robot=robot,algorithm=algorithm,seed=seed,max_delta=max_delta,distance_scale=.1,rate_limit=.1,
                control_frequency=30.,goal_environment=True,environment_version='goal-reach-v1',
                model_version='MuJoCo 3.10.0 / Menagerie c96a32d28fb5da84da38c1da4d749e7a13212855',domain_randomization=domain_randomization,
                task_config=asdict(TaskConfig(success_threshold=target_threshold,success_bonus=success_bonus,max_episode_steps=horizon)),
                evaluation_hold_seconds=hold_seconds,validation_threshold_m=validation_threshold,
                curriculum=curriculum,keep_replay_between_stages=keep_replay_between_stages,observation_schema={'observation':'selected joint q, qvel, end effector XYZ (SI)','achieved_goal':'XYZ metres','desired_goal':'XYZ metres'},
                action_schema={'mode':'normalized_joint_delta','controlled_names':None,'maximum_delta_rad':max_delta,'fixed_omitted_joints':'home','rate_limit_rad_per_step':.1},
                training_seed=seed,stable_baselines3_version=stable_baselines3.__version__,
                sac=dict(buffer_size=buffer_size,batch_size=batch_size,learning_rate=learning_rate,gamma=gamma,tau=tau,train_freq=train_freq,gradient_steps=gradient_steps,learning_starts=learning_starts),
                net_arch=list(net_arch),her=dict(n_sampled_goal=4,goal_selection_strategy='future'))
    state=dict(stage=0 if curriculum else 3,history=[],best_scores={},elapsed_seconds=0.)
    if resume:
        parent=Path(resume).parent
        config=json.loads((parent/'config.json').read_text());state=json.loads((parent/'training_state.json').read_text())
        if config['robot']!=robot:raise ValueError('resume robot mismatch')
        algorithm=config['algorithm'];curriculum=config['curriculum']
    if initialize_from:
        previous=json.loads((Path(initialize_from).parent/'config.json').read_text())
        for key in ('robot','algorithm','max_delta','distance_scale','rate_limit','control_frequency'):
            if config[key]!=previous[key]:raise ValueError(f'warm start {key} mismatch')
        config['net_arch']=previous['net_arch']
        config['initialized_from']=str(initialize_from)
    stages=[dict(x) for x in STAGES]
    stages[-1]['threshold']=config['task_config']['success_threshold']
    stage=stages[state['stage']]
    env=DummyVecEnv([lambda:Monitor(goal_environment(config,stage=stage))])
    env.seed(config['seed'])
    # Training target RNG excludes the held-out sequential seed ranges.
    env.envs[0].unwrapped.reset(seed=config['seed'])
    config['action_schema']['controlled_names']=env.envs[0].unwrapped.actuator_names
    config.update(task='reach',success_threshold_m=config['task_config']['success_threshold'],control_period_s=1/config['control_frequency'],
                  physics_timestep_s=env.envs[0].unwrapped.sim.timestep,action_mode='normalized_joint_delta',
                  training_algorithm={'sac-her':'SAC+HER','sac':'SAC','ppo':'PPO'}[algorithm],
                  internal_units={'joint_position':'radian','joint_velocity':'radian/second','position':'metre','time':'second'})
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    run=None;started=time.monotonic()
    if wandb_enabled:
        import wandb
        run=wandb.init(project='so101-reach-5mm',config=config,dir=str(output),mode=wandb_mode)
    records=[]
    class Metrics(BaseCallback):
        def _on_step(self):
            if self.num_timesteps%100==0:
                metrics={k:float(v) for k,v in self.model.logger.name_to_value.items() if k.startswith('train/') and isinstance(v,(float,int,np.number))}
                metrics['replay_buffer/size']=self.model.replay_buffer.size() if hasattr(self.model,'replay_buffer') else 0
                if run is not None:run.log(metrics,step=self.num_timesteps)
            for done,info in zip(self.locals['dones'],self.locals['infos']):
                if done and 'episode' in info:
                    item=dict(step=self.num_timesteps,reward=info['episode']['r'],length=info['episode']['l'],success=info['is_success'],distance=info['distance'])
                    records.append(item)
                    if run is not None:run.log({'episode/'+k:v for k,v in item.items() if k!='step'},step=self.num_timesteps)
            return True
    try:
        if resume:
            model=(PPO if algorithm=='ppo' else SAC).load(resume,env=env,device='cpu')
            if state.get('training_steps')!=model.num_timesteps:
                raise ValueError('checkpoint and training_state do not match; resume latest.zip or use initialize_from')
            replay=Path(resume).parent/'latest-replay.pkl'
            if algorithm!='ppo':
                if not replay.exists():raise FileNotFoundError('resume requires latest-replay.pkl; use initialize_from for weights only')
                model.load_replay_buffer(replay,truncate_last_traj=True)
            rng=state.get('target_rng_state')
            if rng:env.envs[0].unwrapped.base.np_random.bit_generator.state=rng
            # SAC.load schedules its original seed on VecEnv.reset. Clear it so
            # the saved target RNG is continued instead of silently restarted.
            env._reset_seeds()
            model._last_obs=env.reset()
            model._last_episode_starts=np.ones((1,),dtype=bool)
            if 'numpy_rng_state' in state:
                a,b,c,d,e=state['numpy_rng_state']
                np.random.set_state((a,np.asarray(b,dtype=np.uint32),c,d,e))
            if 'torch_rng_state' in state:torch.set_rng_state(torch.tensor(state['torch_rng_state'],dtype=torch.uint8))
        elif initialize_from:
            # SB3 load otherwise silently retains the source learning schedule.
            overrides={'learning_rate':learning_rate} if algorithm=='ppo' else config['sac']
            model=(PPO if algorithm=='ppo' else SAC).load(initialize_from,env=env,device='cpu',**overrides)
            config['pretrained_steps']=model.num_timesteps
            model.num_timesteps=0
            model.set_random_seed(seed)
            if algorithm!='ppo':model.learning_starts=config['sac']['learning_starts']
            (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        elif algorithm=='ppo':
            model=PPO('MultiInputPolicy',env,n_steps=256,batch_size=64,n_epochs=5,learning_rate=learning_rate,device='cpu',seed=seed,
                      policy_kwargs={'net_arch':dict(pi=list(net_arch),vf=list(net_arch))},verbose=0)
        else:
            kwargs=dict(replay_buffer_class=HerReplayBuffer,replay_buffer_kwargs=config['her']) if algorithm=='sac-her' else {}
            model=SAC('MultiInputPolicy',env,**config['sac'],**kwargs,policy_kwargs={'net_arch':config['net_arch']},device='cpu',seed=seed,verbose=0)
        deadline=min(max_steps,model.num_timesteps+steps)
        while model.num_timesteps<deadline:
            chunk=min(eval_interval,deadline-model.num_timesteps)
            model.learn(chunk,reset_num_timesteps=False,callback=Metrics())
            stage=stages[state['stage']]
            evaluation_count=max(eval_episodes,final_eval_episodes) if state['stage']==3 else eval_episodes
            eval_stage=dict(stage)
            if state['stage']==3:eval_stage['threshold']=config.get('validation_threshold_m',stage['threshold'])
            report=rollout(model,config,episodes=evaluation_count,stage=eval_stage,
                           hold_seconds=config.get('evaluation_hold_seconds',0.) if state['stage']==3 else 0.)
            entry=dict(step=model.num_timesteps,stage=stage['name'],seconds=state['elapsed_seconds']+time.monotonic()-started,
                       evaluation_episodes=evaluation_count,hold_seconds=config.get('evaluation_hold_seconds',0.) if state['stage']==3 else 0.,
                       reach_success_rate=report['reach_success_rate'],success_rate=report['success_rate'],median_distance=report['median_final_distance'],p95_distance=report['percentiles_m']['P95'])
            state['history'].append(entry)
            score=[report['success_rate'],-report['median_final_distance']]
            sample_counts=state.setdefault('best_eval_episodes',{})
            if sample_counts.get(stage['name'])!=evaluation_count:
                state['best_scores'].pop(stage['name'],None)
                sample_counts[stage['name']]=evaluation_count
            if score>state['best_scores'].get(stage['name'],[-1.,-1.]):
                model.save(output/'best');model.save(output/f'best-{stage["name"]}');state['best_scores'][stage['name']]=score
            print(json.dumps(entry),flush=True)
            if run is not None:
                run.log({'evaluation/'+k:v for k,v in entry.items() if isinstance(v,(int,float))}|
                        {'curriculum/stage':state['stage'],'curriculum/success_threshold':stage['threshold'],'curriculum/target_distance_min':stage['minimum'],'curriculum/target_distance_max':stage['maximum']},step=model.num_timesteps)
            advance=curriculum and report['success_rate']>=.9 and state['stage']<3
            if advance:
                state['stage']+=1;new_stage=stages[state['stage']]
                env.envs[0].unwrapped.configure_task(success_threshold=new_stage['threshold'],min_target_distance=new_stage['minimum'],max_target_distance=new_stage['maximum'])
                model._last_obs=None
                # Goal/reward distribution changes. Preserve actor/critic/optimizer,
                # reset replay to avoid mixed-stage relabelling and stale timeouts.
                if algorithm!='ppo' and config.get('keep_replay_between_stages',False):
                    if isinstance(model.replay_buffer,HerReplayBuffer):model.replay_buffer.truncate_last_trajectory()
                if algorithm!='ppo' and not config.get('keep_replay_between_stages',False):
                    klass=HerReplayBuffer if algorithm=='sac-her' else type(model.replay_buffer)
                    kwargs=dict(env=env,**config['her']) if algorithm=='sac-her' else {}
                    model.replay_buffer=klass(config['sac']['buffer_size'],model.observation_space,model.action_space,device='cpu',n_envs=1,**kwargs)
                    # Wait for one complete new-stage episode before HER sampling.
                    model.learning_starts=model.num_timesteps+max(horizon,1000)
                print('advance to '+new_stage['name']+'; weights retained, replay '+('retained' if config.get('keep_replay_between_stages',False) else 'reset'),flush=True)
            model.save(output/'latest')
            if algorithm!='ppo':model.save_replay_buffer(output/'latest-replay.pkl')
            state['training_steps']=model.num_timesteps
            state['target_rng_state']=env.envs[0].unwrapped.base.np_random.bit_generator.state
            a,b,c,d,e=np.random.get_state()
            state['numpy_rng_state']=[a,b.tolist(),c,d,e]
            state['torch_rng_state']=torch.get_rng_state().tolist()
            (output/'training_state.json').write_text(json.dumps(state,indent=2)+'\n')
            (output/'artifact.json').write_text(json.dumps(config|{'training_steps':model.num_timesteps,'current_stage':stages[state['stage']]['name']},indent=2)+'\n')
            with (output/'episodes.jsonl').open('a') as f:
                for item in records:f.write(json.dumps(item)+'\n')
            records.clear()
            if not advance and state['stage']==3 and report['success_rate']>=validation_success:break
        state['elapsed_seconds']+=time.monotonic()-started
        (output/'training_state.json').write_text(json.dumps(state,indent=2)+'\n')
        return state
    finally:
        env.close()
        if run is not None:run.finish()
