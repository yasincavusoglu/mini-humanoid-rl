"""
08_yuruyus_dr_eval.py — Run the ROBUST (Domain Randomization) walking policy
in NOMINAL physics, camera-following video + measure how far it walks.
DR policy trained in randomized physics -> should also walk in nominal (sim-to-real ready).
"""
import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch, torch._dynamo, torch.optim          # BEFORE mujoco (segfault fix)
import sys; sys.path.insert(0, "/home/yasin/Workspace/humanoid_rl")
import mujoco
import imageio.v2 as imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.yuruyus_env import YuruyusEnv

ROOT = "/home/yasin/Workspace/humanoid_rl"
env = YuruyusEnv(render_mode="rgb_array")             # NOMINAL physics
model = PPO.load(f"{ROOT}/models/policy_yuruyus_dr.zip")
vn = VecNormalize.load(f"{ROOT}/models/vecnorm_yuruyus_dr.pkl", DummyVecEnv([lambda: YuruyusEnv()]))
vn.training = False

renderer = mujoco.Renderer(env.model, 640, 480)
cam = mujoco.MjvCamera(); cam.distance, cam.azimuth, cam.elevation = 2.0, 90, -8
obs, _ = env.reset(seed=5)
frames = []
for i in range(1000):
    o = vn.normalize_obs(obs)
    a, _ = model.predict(o, deterministic=True)
    obs, r, term, trunc, _ = env.step(a)
    x = float(env.data.qpos[0]); cam.lookat[:] = [x, 0, 0.25]
    renderer.update_scene(env.data, cam); frames.append(renderer.render())
    if term or trunc:
        break
imageio.mimsave(f"{ROOT}/videos/egitilmis_yuruyus_dr.mp4", frames, fps=50)
print(f"{len(frames)} frames (~{len(frames)*0.02:.1f} s), forward {float(env.data.qpos[0]):.2f} m -> videos/egitilmis_yuruyus_dr.mp4")
