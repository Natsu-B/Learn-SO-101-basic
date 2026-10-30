"""Small SB3 experiment helpers. Imported only by optional learning examples."""
from dataclasses import asdict
from pathlib import Path
import json
import time
from .config import TaskConfig
from .factory import make_env
from .randomization import load_profile, load_parameters


def environment_from_config(config):
    return make_env(config["robot"], task="reach", action_mode=config["action_mode"],
                    max_delta=config["max_delta"], rate_limit=config["rate_limit"],
                    control_frequency=config["control_frequency"],
                    task_config=TaskConfig(**config["task_config"]),
                    domain_randomization=config.get("domain_randomization"),model_parameters=config.get("model_parameters"))


def train(robot, steps, output, *, seed=42, rollout_steps=256, batch_size=64,
          n_epochs=5, task_config=None, action_mode="delta", max_delta=0.05,
          rate_limit=0.05, learning_rate=3e-4, wandb_enabled=False, wandb_mode="online",
          domain_randomization=None, model_parameters=None):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    if steps <= 0 or rollout_steps < 2 or batch_size < 2 or batch_size > rollout_steps or rollout_steps % batch_size:
        raise ValueError("steps must be positive; rollout_steps must be divisible by batch_size >= 2")
    if n_epochs < 1:
        raise ValueError("n_epochs must be positive")
    output = Path(output)
    if (output / "model.zip").exists():
        raise FileExistsError(f"checkpoint already exists: {output / 'model.zip'}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    task_config = task_config or TaskConfig(success_bonus=1.0, max_episode_steps=200)
    config = {"robot": robot, "seed": seed, "action_mode": action_mode,
              "max_delta": max_delta, "rate_limit": rate_limit,
              "domain_randomization": load_profile(domain_randomization),
              "model_parameters": load_parameters(model_parameters),
              "control_frequency": 30.0, "task_config": asdict(task_config),
              "ppo": {"n_steps": rollout_steps, "batch_size": batch_size,
                      "n_epochs": n_epochs, "learning_rate": learning_rate,
                      "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.2}}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    episodes = []
    wandb_run = None

    class EpisodeMetrics(BaseCallback):
        def _on_step(self):
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if done and "episode" in info:
                    episodes.append({"step": self.num_timesteps,
                                     "reward": float(info["episode"]["r"]),
                                     "length": int(info["episode"]["l"]),
                                     "success": bool(info["is_success"]),
                                     "final_distance": float(info["distance"]),
                                     "domain_randomization": info.get("domain_randomization")})
                    if wandb_run is not None:
                        recent=episodes[-100:]
                        parameters=info.get("domain_randomization") or {}
                        dr_metrics={f"domain_randomization/{key}":sum(value)/len(value) if isinstance(value,list) else value
                                    for key,value in parameters.items() if key.endswith("_scale") or key.endswith("_seconds")}

                        wandb_run.log({**dr_metrics,"episode/reward":episodes[-1]["reward"],
                                       "episode/length":episodes[-1]["length"],
                                       "episode/final_distance":episodes[-1]["final_distance"],
                                       "rollout/success_rate":sum(e["success"] for e in recent)/len(recent),
                                       "train/learning_rate":self.model.policy.optimizer.param_groups[0]["lr"]},
                                      step=self.num_timesteps)
            return True

    env = DummyVecEnv([lambda: Monitor(environment_from_config(config))])
    started = time.monotonic()
    try:
        if wandb_enabled:
            import wandb
            wandb_run=wandb.init(project="robot-sim-reach",config=config,dir=str(output),mode=wandb_mode)
        model = PPO("MultiInputPolicy", env, device="cpu", seed=seed, verbose=0,
                    policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
                    **config["ppo"])
        model.learn(total_timesteps=steps, callback=EpisodeMetrics())
        model.save(output / "model")
        result = {"robot": robot, "requested_steps": steps,
                  "training_steps": model.num_timesteps, "optimizer_updates": model._n_updates,
                  "seconds": time.monotonic() - started, "episodes": episodes,
                  "checkpoint": str(output / "model.zip")}
        (output / "training.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()


def evaluate(checkpoint, *, robot=None, episodes=20, seed=10000, rerun=False, record=None, domain_randomization=None):
    """Compare policies on identical held-out target seeds; telemetry records trained episodes."""
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    if episodes < 1:
        raise ValueError("episodes must be positive")
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "model.zip"
    config = json.loads((checkpoint.parent / "config.json").read_text())
    if robot is not None and robot != config["robot"]:
        raise ValueError("robot does not match the checkpoint configuration")
    if domain_randomization is not None:
        config["domain_randomization"]=load_profile(domain_randomization)
    torch.set_num_threads(1)
    model = PPO.load(checkpoint, device="cpu")
    result = {"robot": config["robot"], "training_steps": model.num_timesteps,
              "episodes": episodes, "first_seed": seed,
              "domain_randomization": config.get("domain_randomization",{}),
              "model_parameters": config.get("model_parameters")}
    for policy in ("random", "trained"):
        env = environment_from_config(config)
        outcomes = []
        try:
            if policy == "trained" and (rerun or record is not None):
                env.unwrapped.sim.start_rerun(record=str(record) if record is not None else None)
            for index in range(episodes):
                obs, info = env.reset(seed=seed + index)
                env.action_space.seed(seed + index)
                total = 0.0
                for length in range(1, config["task_config"]["max_episode_steps"] + 1):
                    action = env.action_space.sample() if policy == "random" else model.predict(obs, deterministic=True)[0]
                    obs, reward, terminated, truncated, info = env.step(action)
                    total += reward
                    if terminated or truncated:
                        break
                outcomes.append({"seed": seed + index, "reward": total, "length": length,
                                 "success": bool(info["is_success"]), "final_distance": info["distance"]})
            if policy == "trained" and (rerun or record is not None):
                env.unwrapped.sim.flush_telemetry()
        finally:
            env.close()
        result[policy] = {"success_rate": float(np.mean([e["success"] for e in outcomes])),
                          "mean_episode_reward": float(np.mean([e["reward"] for e in outcomes])),
                          "mean_final_distance": float(np.mean([e["final_distance"] for e in outcomes])),
                          "mean_episode_length": float(np.mean([e["length"] for e in outcomes])),
                          "outcomes": outcomes}
    return result
