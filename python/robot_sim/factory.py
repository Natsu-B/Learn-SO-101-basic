from .env import RobotEnv
from .wrappers import DeltaJointPositionWrapper, NormalizedJointPositionWrapper, RateLimitWrapper


def make_env(robot: str, task: str = "reach", seed: int | None = None, *,
             action_mode="delta", max_delta=0.05, rate_limit=None,
             task_config=None, control_frequency=30.0, domain_randomization=None, model_parameters=None):
    """One independent worker per environment, suitable for SyncVectorEnv.

    Vectorize environments with the same observation/action shape together.
    Panda (7 joints) cannot share a homogeneous vector batch with 6-joint arms.
    """
    if action_mode not in {"absolute", "normalized", "delta"}:
        raise ValueError("action_mode must be absolute, normalized or delta")
    env = RobotEnv(robot=robot, task=task, task_config=task_config,
                   control_frequency=control_frequency, domain_randomization=domain_randomization,
                   model_parameters=model_parameters)
    try:
        if rate_limit is not None:
            env = RateLimitWrapper(env, max_change=rate_limit)
        if action_mode == "delta":
            env = DeltaJointPositionWrapper(env, max_delta=max_delta)
        elif action_mode == "normalized":
            env = NormalizedJointPositionWrapper(env)
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
        return env
    except Exception:
        env.close()
        raise
