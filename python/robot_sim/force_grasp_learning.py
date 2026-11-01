"""PPO training and evaluation for the grasp-force environment (docs/FORCE_CONTROL.md §6–7).

PPO settings follow the paper's public training scripts where possible
(SB3 PPO, MlpPolicy [50, 50], lr 6e-4, 150-step episodes, 3 stacked frames,
curriculum until a third of training); deviations are recorded in the
saved ``config.json``. Checkpoint selection uses SELECTION seeds, final
comparisons FINAL seeds; training episodes draw seeds from the RNG.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .force_grasp_env import (
    ForceGraspConfig,
    ForceGraspEnv,
    fixed_command_controller,
    pi_controller,
)

SELECTION_SEEDS = 7_000_000
FINAL_SEEDS = 8_000_000
SUMMARY_KEYS = (
    "success", "mean_abs_error", "max_abs_error", "relative_error", "peak_force_physics",
    "peak_over_goal", "crushed", "invalid_contact", "lost_contact_final", "both_contact_ratio",
    "object_displacement_mm", "return", "action_saturation",
)


def run_episode(env: ForceGraspEnv, seed: int, controller: str, model=None,
                kappa: float | None = None) -> dict:
    """One episode with ``controller`` in fixed | pi | ppo; returns metrics."""
    from ._native import ForcePi

    obs, _ = env.reset(options={"episode_seed": seed, "kappa": kappa})
    act: Callable | None = None
    if controller == "fixed":
        act = fixed_command_controller(env)
    elif controller == "pi":
        act = pi_controller(env, ForcePi())
    elif controller != "ppo":
        raise ValueError(f"unknown controller {controller}")
    while True:
        if act is None:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
        else:
            obs, _, terminated, truncated, info = env.step_command(act())
        if terminated or truncated:
            break
    m = info.get("episode_metrics") or env.metrics.summary(
        env.config.contact_threshold, env.penetration_limit)
    m.update(seed=seed, controller=controller, error=info.get("error"), object=env.object,
             terminated=bool(terminated), precontact=env.precontact, kappa=kappa)
    return m


def evaluate(env: ForceGraspEnv, seeds: Iterable[int], controller: str, model=None) -> list[dict]:
    env.set_progress(1.0)
    return [run_episode(env, int(s), controller, model) for s in seeds]


def aggregate(rows: list[dict]) -> dict:
    out = {"episodes": len(rows)}
    for k in SUMMARY_KEYS:
        v = np.asarray([float(r[k]) for r in rows], dtype=float)
        out[k] = float(np.nanmean(v)) if len(v) else float("nan")
    peaks = np.asarray([r["peak_force_physics"] for r in rows])
    out["peak_force_max"] = float(peaks.max()) if len(peaks) else float("nan")
    # Peaks of episodes whose contacts stayed geometrically valid (no plate
    # tunnelling into a pad): the physically meaningful impact force.
    valid = [r for r in rows if not r["invalid_contact"]]
    out["valid_episodes"] = len(valid)
    vp = np.asarray([r["peak_force_physics"] for r in valid])
    out["peak_force_valid_mean"] = float(vp.mean()) if len(vp) else float("nan")
    out["peak_force_valid_median"] = float(np.median(vp)) if len(vp) else float("nan")
    out["peak_force_valid_max"] = float(vp.max()) if len(vp) else float("nan")
    out["peak_over_goal_valid"] = (
        float(np.mean([r["peak_over_goal"] for r in valid])) if valid else float("nan"))
    for phase in ("push", "contact", "hold", "none"):
        out[f"peak_phase_{phase}"] = sum(r.get("peak_phase") == phase for r in rows)
    out["peak_force_hold"] = float(np.mean([r.get("peak_force_hold", np.nan) for r in rows]))
    t = [r["time_to_stable_s"] for r in rows if r["time_to_stable_s"] is not None]
    out["stable_reached"] = len(t)
    out["time_to_stable_mean_s"] = float(np.mean(t)) if t else None
    return out


def _make_env(config_dict: dict, seed: int):
    def init():
        env = ForceGraspEnv(ForceGraspConfig.from_dict(config_dict))
        env.reset(seed=seed)
        return env
    return init


def train_force_grasp(output: str, *, steps: int, seed: int = 42, n_envs: int = 6,
                      n_steps: int = 1024, batch_size: int = 64, n_epochs: int = 10,
                      learning_rate: float = 6e-4, curriculum_fraction: float = 1 / 3,
                      eval_freq: int = 10_000, eval_episodes: int = 24,
                      checkpoint_freq: int = 10_000, config: ForceGraspConfig | None = None,
                      resume: str | None = None) -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} exists and is not empty; refusing to overwrite")
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    config = config or ForceGraspConfig()
    config.validate()
    curriculum_end = max(1, int(curriculum_fraction * steps))
    ppo_kwargs = dict(learning_rate=learning_rate, n_steps=n_steps, batch_size=batch_size,
                      n_epochs=n_epochs, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                      ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                      policy_kwargs=dict(net_arch=[50, 50]))
    meta = {
        "steps": steps, "seed": seed, "n_envs": n_envs, "curriculum_end": curriculum_end,
        "eval_freq": eval_freq, "eval_episodes": eval_episodes,
        "selection_seeds": [SELECTION_SEEDS, SELECTION_SEEDS + eval_episodes - 1],
        "algorithm": "PPO", "policy": "MlpPolicy", "ppo": ppo_kwargs, "env": config.to_dict(),
        "resume": resume,
        "deviations_from_paper_code": [
            f"{n_envs} parallel envs (paper code: 10; M1 runs at most 6 well)",
            f"n_steps {n_steps} per env (paper code: SB3 default 2048)",
            "stock SB3 2.x (paper code: a modified SB3 fork)",
            "30 Hz control (paper: 25 Hz), one moving jaw (paper: two fingers)",
            "inductive bias adapted to one moving jaw (docs/FORCE_CONTROL.md §1.2)",
            "r_obj active with the paper's alpha2 curriculum (public scripts: weight 0)",
            "r_force normalized by f_goal as in the public code (paper eq. 8: not normalized)",
        ],
    }
    (out / "config.json").write_text(json.dumps(meta, indent=2))
    cfg = config.to_dict()
    venv = VecMonitor(SubprocVecEnv([_make_env(cfg, seed + i) for i in range(n_envs)]))
    if resume:
        model = PPO.load(resume, env=venv, device="cpu")
    else:
        model = PPO("MlpPolicy", venv, seed=seed, device="cpu", verbose=0, **ppo_kwargs)
    model.set_logger(configure(str(out), ["csv"]))
    eval_env = ForceGraspEnv(ForceGraspConfig.from_dict(cfg))
    selection = list(range(SELECTION_SEEDS, SELECTION_SEEDS + eval_episodes))
    log_path = out / "selection.jsonl"
    started = time.time()

    class Callback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.best = -np.inf
            self.next_eval = eval_freq
            self.next_ckpt = checkpoint_freq
            self.recent: list[dict] = []
            self.sim_time = 0.0

        def _on_training_start(self) -> None:
            # Resumed runs start at the loaded model's timestep count.
            self.next_eval = self.num_timesteps + eval_freq
            self.next_ckpt = self.num_timesteps + checkpoint_freq

        def _on_rollout_start(self) -> None:
            progress = min(1.0, self.num_timesteps / curriculum_end)
            self.training_env.env_method("set_progress", progress)
            self.logger.record("curriculum/progress", progress)

        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                m = info.get("episode_metrics")
                if m:
                    self.recent.append(m)
            if self.num_timesteps >= self.next_ckpt:
                self.model.save(out / "checkpoints" / f"step_{self.num_timesteps}.zip")
                self.next_ckpt += checkpoint_freq
            if self.num_timesteps >= self.next_eval:
                self.next_eval += eval_freq
                self._evaluate()
            return True

        def _on_rollout_end(self) -> None:
            if self.recent:
                agg = aggregate(self.recent[-100:])
                for k in ("success", "relative_error", "peak_force_physics", "peak_over_goal",
                          "crushed", "both_contact_ratio", "action_saturation"):
                    self.logger.record(f"train_episode/{k}", agg[k])
            self.logger.record("time/wall_s", time.time() - started)

        def _evaluate(self) -> None:
            t = time.time()
            rows = evaluate(eval_env, selection, "ppo", self.model)
            agg = aggregate(rows)
            agg.update(timesteps=self.num_timesteps, wall_s=time.time() - started,
                       eval_s=time.time() - t)
            with log_path.open("a") as f:
                f.write(json.dumps(agg) + "\n")
            for k in ("return", "success", "relative_error", "peak_force_physics", "crushed"):
                self.logger.record(f"selection/{k}", agg[k])
            if agg["return"] > self.best:
                self.best = agg["return"]
                self.model.save(out / "best.zip")
                (out / "best.json").write_text(json.dumps(agg, indent=2))

    cb = Callback()
    model.learn(total_timesteps=steps, callback=cb, reset_num_timesteps=resume is None)
    cb._evaluate()
    model.save(out / "final.zip")
    wall = time.time() - started
    summary = {"output": str(out), "timesteps": int(model.num_timesteps), "wall_s": wall,
               "steps_per_s": steps / wall, "best_selection_return": cb.best}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    venv.close()
    eval_env.close()
    return summary
