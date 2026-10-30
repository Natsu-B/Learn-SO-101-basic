"""Rust simulator and optional Gymnasium environment."""
from ._native import (
    Simulator, PickPlaceSimulator, SineTrajectory, ACTUATOR_NAMES, __version__,
    GRIPPER_OPEN, GRIPPER_CLOSED, CONTROL_PERIOD,
)
from .config import TaskConfig

def __getattr__(name):
    if name == "GoalTransportEnv":
        from .transport_env import GoalTransportEnv
        return GoalTransportEnv
    if name == "GoalReachEnv":
        from .goal_env import GoalReachEnv
        return GoalReachEnv
    if name == "make_env":
        from .factory import make_env
        return make_env
    if name == "RobotEnv":
        from .env import RobotEnv
        return RobotEnv
    if name in {"NormalizedJointPositionWrapper", "DeltaJointPositionWrapper", "RateLimitWrapper", "ResidualActionWrapper"}:
        from . import wrappers
        return getattr(wrappers, name)
    raise AttributeError(name)

__all__ = [
    "Simulator", "PickPlaceSimulator", "RobotEnv", "ACTUATOR_NAMES",
    "GRIPPER_OPEN", "GRIPPER_CLOSED", "CONTROL_PERIOD",
]
