"""Small SAC+HER runner for transport; Reach's curriculum is left unchanged."""
from pathlib import Path
import json
import time

import gymnasium as gym
import numpy as np

from .transport_env import GoalTransportEnv
from ._native import CONTROL_PERIOD


def evaluate_transport(model, config, *, episodes=20, seed=1000000, emit=None, env=None):
    """Deterministic inference on sequential native seeds, stopping on Rust predicates."""
    if episodes < 1 or seed < 0:
        raise ValueError("episodes must be positive and seed non-negative")
    owns_env = env is None
    if owns_env:
        env = GoalTransportEnv(**config["env"], terminate_on_success=True, terminate_on_drop=True)
    outcomes = []
    try:
        for requested_seed in range(seed, seed + episodes):
            obs, info = env.reset(seed=requested_seed)
            grasp = info["grasp_report"]
            dropped = False
            for step in range(1, env.horizon + 1):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                dropped |= info["dropped"]
                if terminated or truncated:
                    break
            row = dict(type="episode", requested_seed=requested_seed, seed=info["seed"],
                       grasp_retries=info["grasp_retries"], success=info["is_success"],
                       dropped=dropped, steps=step, final_xy_error_m=info["xy_error"],
                       final_goal_distance_m=info["distance"],
                       cube_final=env.last_result["cube_pos"], target=env.last_result["target"],
                       skills=[grasp, dict(name="transport", learned=True, success=info["is_success"],
                                           steps=step, reason=None if info["is_success"] else
                                           ("dropped" if dropped else "timeout"))])
            outcomes.append(row)
            if emit is not None:
                emit(row)
        errors = np.asarray([r["final_xy_error_m"] for r in outcomes])
        return dict(type="summary", episodes=episodes, seed=seed,
                    success_rate=float(np.mean([r["success"] for r in outcomes])),
                    drop_rate=float(np.mean([r["dropped"] for r in outcomes])),
                    final_xy_error_m=dict(mean=float(errors.mean()), p95=float(np.percentile(errors, 95)),
                                          max=float(errors.max())),
                    grasp_success_rate=(env.grasp_attempts_total - env.grasp_failures_total)
                                       / env.grasp_attempts_total,
                    grasp_attempts=env.grasp_attempts_total, grasp_failures=env.grasp_failures_total,
                    learned_skills=["transport"], outcomes=outcomes)
    finally:
        if owns_env:
            env.close()


class _TimedEnv(gym.Wrapper):
    """Separate physics/reset throughput from inference, optimizer and evaluation time."""
    def __init__(self, env):
        super().__init__(env)
        self.steps = 0
        self.step_seconds = 0.
        self.reset_seconds = 0.

    def reset(self, **kwargs):
        started = time.perf_counter()
        result = self.env.reset(**kwargs)
        self.reset_seconds += time.perf_counter() - started
        return result

    def step(self, action):
        started = time.perf_counter()
        result = self.env.step(action)
        self.step_seconds += time.perf_counter() - started
        self.steps += 1
        return result


def train_transport(output, *, steps=250000, seed=42, resume=None, horizon=150,
                    max_delta=.05, rate_limit=.1, distance_scale=.1,
                    eval_interval=10000, eval_episodes=20, eval_seed=1000000,
                    buffer_size=300000, batch_size=256, learning_rate=.001,
                    train_freq=16, gradient_steps=4, learning_starts=1000):
    """Same SAC/HER defaults as train_goal; save weights, replay and RNG state.

    Resume latest.zip/final.zip together with their matching latest-replay.pkl
    and training_state.json. Configuration is restored from the source directory;
    steps is an additional budget. This starts a new physical episode on resume.
    """
    import torch
    import stable_baselines3
    from stable_baselines3 import SAC, HerReplayBuffer
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    if min(steps, eval_interval, eval_episodes, train_freq, gradient_steps, batch_size) < 1:
        raise ValueError("training and evaluation budgets must be positive")
    if learning_starts < 0:
        raise ValueError("learning_starts must be non-negative")
    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if resume is None and (output / "config.json").exists():
        raise FileExistsError("use --resume or choose a new output directory")
    config = dict(task="transport", environment_version="goal-transport-v1", seed=seed,
                  algorithm="SAC+HER", stable_baselines3_version=stable_baselines3.__version__,
                  env=dict(horizon=horizon, max_delta=max_delta, rate_limit=rate_limit,
                           distance_scale=distance_scale),
                  reward="-norm(achieved_goal - desired_goal) / distance_scale",
                  observation_schema="arm q[5], arm qvel[5], grasp_center[3], cube_pos[3], "
                                     "cube_linear_velocity[3], gripper[1], cube-grasp_center[3]",
                  action_schema="measured arm q + normalized delta; slew limited against previous command; gripper closed",
                  sac=dict(buffer_size=buffer_size, batch_size=batch_size, learning_rate=learning_rate,
                           gamma=.98, tau=.005, train_freq=train_freq, gradient_steps=gradient_steps,
                           learning_starts=max(learning_starts, horizon)),
                  net_arch=[256, 256, 256], her=dict(n_sampled_goal=4, goal_selection_strategy="future"))
    previous_state = None
    if resume is not None:
        source = Path(resume).parent
        config = json.loads((source / "config.json").read_text())
        previous_state = json.loads((source / "training_state.json").read_text())
        if config.get("environment_version") != "goal-transport-v1":
            raise ValueError("resume requires a transport checkpoint")
    if config["sac"]["buffer_size"] <= config["env"]["horizon"]:
        raise ValueError("HER buffer_size must exceed the episode horizon")
    timed = _TimedEnv(GoalTransportEnv(**config["env"]))
    env = DummyVecEnv([lambda: Monitor(timed)])
    env.seed(config["seed"])
    config.update(criteria=timed.unwrapped.criteria, control_period_s=CONTROL_PERIOD,
                  physics_timestep_s=timed.unwrapped.sim.timestep(),
                  evaluation=dict(seed=eval_seed, episodes=eval_episodes, interval=eval_interval))
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    best_score = [-1., -float("inf")]
    if (previous_state is not None and Path(resume).parent.resolve() == output.resolve()
            and previous_state.get("evaluation_signature") == [eval_seed, eval_episodes]):
        best_score = previous_state["best_score"]
    history = []
    started = time.perf_counter()
    evaluation_seconds = 0.

    def checkpoint(model):
        model.save(output / "latest")
        model.save_replay_buffer(output / "latest-replay.pkl")
        a, b, c, d, e = np.random.get_state()
        state = dict(training_steps=model.num_timesteps, best_score=best_score,
                     evaluation_signature=[eval_seed, eval_episodes],
                     env_rng_state=timed.unwrapped.np_random.bit_generator.state,
                     numpy_rng_state=[a, b.tolist(), c, d, e], torch_rng_state=torch.get_rng_state().tolist())
        (output / "training_state.json").write_text(json.dumps(state, indent=2) + "\n")

    class EvaluationCallback(BaseCallback):
        last_eval_step = -1

        def evaluate(self):
            nonlocal best_score, evaluation_seconds
            eval_started = time.perf_counter()
            report = evaluate_transport(self.model, config, episodes=eval_episodes, seed=eval_seed)
            report["training_steps"] = self.model.num_timesteps
            score = [report["success_rate"], -report["final_xy_error_m"]["mean"]]
            if score > best_score:
                best_score = score
                self.model.save(output / "best")
            with (output / "evaluations.jsonl").open("a") as f:
                f.write(json.dumps(report) + "\n")
            history.append({k: v for k, v in report.items() if k != "outcomes"})
            print(json.dumps(history[-1]), flush=True)
            checkpoint(self.model)
            self.last_eval_step = self.model.num_timesteps
            evaluation_seconds += time.perf_counter() - eval_started

        def _on_step(self):
            if self.n_calls % eval_interval == 0:
                self.evaluate()
            return True

        def _on_training_end(self):
            if self.last_eval_step != self.model.num_timesteps:
                self.evaluate()

    try:
        if resume is not None:
            model = SAC.load(resume, env=env, device="cpu")
            if model.num_timesteps != previous_state["training_steps"]:
                raise ValueError("checkpoint and replay state differ; resume latest.zip or final.zip")
            model.load_replay_buffer(Path(resume).parent / "latest-replay.pkl", truncate_last_traj=True)
            timed.unwrapped.np_random.bit_generator.state = previous_state["env_rng_state"]
            env._reset_seeds()  # SAC.load schedules the original seed; continue saved goal RNG instead.
            a, b, c, d, e = previous_state["numpy_rng_state"]
            np.random.set_state((a, np.asarray(b, dtype=np.uint32), c, d, e))
            torch.set_rng_state(torch.tensor(previous_state["torch_rng_state"], dtype=torch.uint8))
        else:
            model = SAC("MultiInputPolicy", env, **config["sac"], replay_buffer_class=HerReplayBuffer,
                        replay_buffer_kwargs=config["her"], policy_kwargs={"net_arch": config["net_arch"]},
                        device="cpu", seed=config["seed"], verbose=0)
        initial_steps = model.num_timesteps
        initial_updates = model._n_updates
        model.learn(steps, reset_num_timesteps=False, callback=EvaluationCallback())
        model.save(output / "final")
        checkpoint(model)
        elapsed = time.perf_counter() - started
        env_seconds = timed.step_seconds + timed.reset_seconds
        result = dict(type="training_summary", training_steps=model.num_timesteps,
                      steps_this_run=model.num_timesteps - initial_steps,
                      optimizer_updates_this_run=model._n_updates - initial_updates,
                      elapsed_seconds=elapsed, evaluation_seconds=evaluation_seconds,
                      training_steps_per_second=timed.steps / (elapsed - evaluation_seconds),
                      env_steps_per_second=timed.steps / env_seconds,
                      env_step_seconds=timed.step_seconds, env_reset_seconds=timed.reset_seconds,
                      grasp_attempts=timed.unwrapped.grasp_attempts_total,
                      grasp_failures=timed.unwrapped.grasp_failures_total,
                      final_model=str(output / "final.zip"), best_model=str(output / "best.zip"),
                      evaluations=history)
        (output / "training_summary.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        env.close()
