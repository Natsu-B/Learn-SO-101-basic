from robot_sim import RobotEnv

env = RobotEnv(robot="so101", task="reach")
obs, info = env.reset(seed=42)
action = obs["joint_position"].copy()
for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(action)
    last_info = info
    if terminated or truncated:
        obs, info = env.reset()
print(last_info)
env.close()
