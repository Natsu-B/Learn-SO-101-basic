"""Headless simulation using the same Rust core as so101-cli."""
from so101_mujoco import Simulator

sim = Simulator()
print("actuators:", sim.actuator_names)
print("initial:", sim.state())
for _ in range(1000):
    state = sim.step([0.0] * 6)
print("time:", sim.time, "state:", state)
sim.reset()
print("reset:", sim.state())
