from dataclasses import dataclass

@dataclass(frozen=True)
class TaskConfig:
    success_threshold: float = 0.02
    success_bonus: float = 0.0
    max_episode_steps: int = 1000
    min_target_distance: float = 0.05
    max_target_distance: float = 0.25
    joint_margin: float = 0.05
    max_target_attempts: int = 4096
