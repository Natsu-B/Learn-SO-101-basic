"""The Rust SineController's signal, applied through the shared simulator."""
import math
from so101_mujoco import Simulator

sim = Simulator()
for _ in range(1000):
    command = [0.0] * 6
    command[0] = 0.3 * math.sin(math.tau * 0.25 * sim.time)
    state = sim.step(command)
print("time:", sim.time, "state:", state)
