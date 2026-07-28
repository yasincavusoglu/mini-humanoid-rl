"""
04_env_test.py — Test the balance GAME (no TRAINING yet, we play randomly).
Goal: see that the environment works + record the pushed-and-flailing robot to video.
After training we will compare against this.
"""
import os, sys
sys.path.insert(0, "/home/yasin/Workspace/humanoid_rl")
os.environ.setdefault("MUJOCO_GL", "osmesa")
import imageio.v2 as imageio
from envs.denge_env import DengeEnv

env = DengeEnv(render_mode="rgb_array")
obs, _ = env.reset(seed=0)
print("Observation size :", obs.shape, " | Action size:", env.action_space.shape)

frames, toplam, bolum = [], 0.0, 1
for i in range(300):
    a = env.action_space.sample()          # RANDOM command (no training)
    obs, r, term, trunc, _ = env.step(a)
    toplam += r
    frames.append(env.render())
    if term or trunc:
        sebep = "FELL" if term else "time up"
        print(f"  episode {bolum}: {sebep} @step {i+1} | total reward = {toplam:6.1f}")
        obs, _ = env.reset(); toplam = 0.0; bolum += 1

imageio.mimsave("/home/yasin/Workspace/humanoid_rl/videos/5_denge_rastgele.mp4", frames, fps=50)
print("Video -> videos/5_denge_rastgele.mp4  (random = flailing and falling)")
