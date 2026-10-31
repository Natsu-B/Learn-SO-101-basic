"""SAC runner for the residual Grasp (with or without observation history).

Training seeds come from Gymnasium's generator seeded by ``seed``; evaluation
uses a disjoint fixed seed block and fixed (mass, friction) conditions, so the
best checkpoint is selected without touching the final evaluation seeds.
"""
from pathlib import Path
import json
import time

import numpy as np

from .grasp_env import GoalGraspEnv

ENV_VERSION = "goal-grasp-residual-v1"
# Checkpoint-selection conditions, fixed before training: the nominal range
# plus the slippery regime where the scripted Grasp was measured to fail. They
# use their own seed block, disjoint from the final evaluation seeds.
SELECTION_CONDITIONS = [(0.035, 1.15), (0.05, 0.8), (0.03, 0.1), (0.07, 0.07), (0.05, 0.15)]


class FixedResidualPolicy:
    """Non-learned baseline: the IK Grasp plus a constant joint offset (rad).

    Uses the same residual path as the learned policy; ``learned`` is False
    so reports keep it separate from AI skills."""

    learned = False

    def __init__(self, offset, *, gripper=0.0, horizon=75, gripper_scale=0.35):
        offset = np.asarray(offset, np.float64)
        if offset.shape != (5,) or not np.isfinite(offset).all():
            raise ValueError("offset must be five finite joint offsets")
        self.arm_scale = max(float(np.abs(offset).max()), 1e-9)
        self.action = np.r_[offset / self.arm_scale, gripper / gripper_scale].astype(np.float32)
        if np.abs(self.action).max() > 1:
            raise ValueError("gripper offset exceeds gripper_scale")
        self.config = dict(history=0, horizon=horizon, arm_scale=self.arm_scale,
                           gripper_scale=gripper_scale)

    def predict(self, obs, deterministic=True):
        return self.action.copy(), None


def wrist_roll_policy(radians):
    """Constant wrist_roll correction (joint index 4)."""
    return FixedResidualPolicy([0, 0, 0, 0, radians])


def evaluate_grasp(model, env, *, conditions, seeds, emit=None):
    """Deterministic residual policy on fixed conditions; returns rates by condition."""
    rows = []
    for mass, friction in conditions:
        env.set_physics_ranges((mass, mass), (friction, friction))
        for seed in seeds:
            obs, info = env.reset(seed=seed)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            row = dict(type="episode", mass=mass, friction=friction, seed=info["seed"],
                       success=info["is_success"], dropped=info["dropped"],
                       steps=info["episode_step"], max_slip_m=info["max_slip"])
            rows.append(row)
            if emit is not None:
                emit(row)
    return dict(success_rate=float(np.mean([r["success"] for r in rows])),
                drop_rate=float(np.mean([r["dropped"] for r in rows])),
                mean_steps=float(np.mean([r["steps"] for r in rows])),
                mean_max_slip_mm=float(np.mean([r["max_slip_m"] for r in rows]) * 1e3),
                episodes=len(rows), outcomes=rows)


def load_grasp_policy(path):
    """Return (model, env_config) for a saved checkpoint; no env is needed."""
    import torch
    from stable_baselines3 import SAC

    torch.set_num_threads(1)
    path = Path(path)
    config = json.loads((path.parent / "config.json").read_text())
    if config.get("environment_version") != ENV_VERSION:
        raise ValueError(f"{path} is not a {ENV_VERSION} checkpoint")
    return SAC.load(path, device="cpu"), config["env"]


def train_grasp(output, *, steps=60000, seed=42, history=0, horizon=75, arm_scale=0.06,
                gripper_scale=0.35, mass_range=(0.02, 0.1), friction_range=(0.05, 1.5),
                friction_sampling="log",
                eval_interval=5000, eval_episodes=6, eval_seed=4000000,
                buffer_size=200000, batch_size=256, learning_rate=3e-4,
                train_freq=8, gradient_steps=4, learning_starts=1000):
    import torch
    import stable_baselines3
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "config.json").exists():
        raise FileExistsError("choose a new output directory")
    env_config = dict(history=history, horizon=horizon, arm_scale=arm_scale,
                      gripper_scale=gripper_scale, mass_range=list(mass_range),
                      friction_range=list(friction_range), friction_sampling=friction_sampling)
    config = dict(task="grasp", environment_version=ENV_VERSION, seed=seed, algorithm="SAC",
                  stable_baselines3_version=stable_baselines3.__version__, env=env_config,
                  hidden_physics="mass and friction randomized at reset; not observed",
                  base_controller="scripted IK Grasp (Close/Lift/Hold) proposes a_IK each step",
                  action="a_IK + [arm_scale*a[0:5], gripper_scale*a[5]], clipped to bounds",
                  success="Rust lifted predicate held for hold_seconds",
                  sac=dict(buffer_size=buffer_size, batch_size=batch_size,
                           learning_rate=learning_rate, gamma=0.98, tau=0.005,
                           train_freq=train_freq, gradient_steps=gradient_steps,
                           learning_starts=learning_starts),
                  net_arch=[256, 256],
                  evaluation=dict(seed=eval_seed, episodes=eval_episodes,
                                  interval=eval_interval, conditions=SELECTION_CONDITIONS))
    env = Monitor(GoalGraspEnv(**env_config))
    eval_env = GoalGraspEnv(**{**env_config, "friction_sampling": "uniform"})
    config["criteria"] = env.unwrapped.criteria
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env.reset(seed=seed)
    best = [-1.0, float("inf")]
    history_rows = []
    started = time.perf_counter()
    eval_seconds = 0.0
    seeds = range(eval_seed, eval_seed + eval_episodes)

    class Callback(BaseCallback):
        def evaluate(self):
            nonlocal best, eval_seconds
            t0 = time.perf_counter()
            report = evaluate_grasp(self.model, eval_env, conditions=SELECTION_CONDITIONS,
                                    seeds=seeds)
            report.pop("outcomes")
            report.update(training_steps=self.model.num_timesteps,
                          wall_seconds=time.perf_counter() - started)
            score = [report["success_rate"], report["mean_max_slip_mm"]]
            if score[0] > best[0] or (score[0] == best[0] and score[1] < best[1]):
                best = score
                self.model.save(output / "best")
                report["saved_best"] = True
            self.model.save(output / "latest")
            history_rows.append(report)
            with (output / "evaluations.jsonl").open("a") as f:
                f.write(json.dumps(report) + "\n")
            print(json.dumps(report), flush=True)
            eval_seconds += time.perf_counter() - t0

        def _on_step(self):
            if self.n_calls % eval_interval == 0:
                self.evaluate()
            return True

    try:
        model = SAC("MlpPolicy", env, **config["sac"], policy_kwargs={"net_arch": config["net_arch"]},
                    device="cpu", seed=seed, verbose=0)
        callback = Callback()
        model.learn(steps, callback=callback)
        model.save(output / "final")
        if not history_rows or history_rows[-1]["training_steps"] != model.num_timesteps:
            callback.evaluate()
        elapsed = time.perf_counter() - started
        summary = dict(type="training_summary", training_steps=model.num_timesteps,
                       elapsed_seconds=elapsed, evaluation_seconds=eval_seconds,
                       training_steps_per_second=model.num_timesteps / (elapsed - eval_seconds),
                       optimizer_updates=model._n_updates, best_score=best,
                       episodes=len(env.get_episode_rewards()),
                       best_model=str(output / "best.zip"), final_model=str(output / "final.zip"))
        (output / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    finally:
        env.close()
        eval_env.close()
