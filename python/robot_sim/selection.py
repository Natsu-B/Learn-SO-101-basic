"""Task-specific controlled subset; RobotSpec/native actuator vector remains intact."""
import gymnasium as gym
import numpy as np
from .wrappers import _action

class ActuatedJointSelection(gym.Wrapper):
    def __init__(self,env,names):
        super().__init__(env)
        all_names=env.unwrapped.sim.actuator_names
        if not names or len(set(names))!=len(names) or any(n not in all_names for n in names):
            raise ValueError('invalid actuator selection')
        self.indices=np.asarray([all_names.index(n) for n in names])
        self.fixed=np.asarray(env.unwrapped.sim.home_position,np.float64)
        env.unwrapped.sim.configure_reach_selection(list(names))
        self.action_space=gym.spaces.Box(env.action_space.low[self.indices],env.action_space.high[self.indices],dtype=np.float32)

    def step(self,action):
        values=_action(action,self.action_space)
        full=self.fixed.copy();full[self.indices]=values
        return self.env.step(full)


def reach_actuators(robot,all_names):
    # Preset-specific task choice, not a fixed dimension in core.
    return [n for n in all_names if not (robot=='so101' and n=='gripper')]
