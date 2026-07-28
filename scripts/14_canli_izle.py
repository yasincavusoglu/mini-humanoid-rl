"""
14_canli_izle.py — Watch a trained policy LIVE in a 3D window (real-time, rotate/zoom).
Usage (in YOUR OWN terminal, GPU required -> OK after reboot):
    DISPLAY=:1 .venv/bin/python scripts/14_canli_izle.py --pol yuru9
Window controls: left-click=rotate, right-click=pan, wheel=zoom, double-click+Ctrl-right-drag=push.
Resets automatically when the robot falls. To close: close the window / Ctrl+C.
"""
import os
os.environ.setdefault("MUJOCO_GL", "glfw")   # for the SCREEN (GPU) -- NOT osmesa
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch, torch._dynamo, torch.optim      # BEFORE mujoco
import sys, argparse, time
sys.path.insert(0, "/home/yasin/Workspace/humanoid_rl")
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ap = argparse.ArgumentParser(); ap.add_argument("--pol", default="yuru9"); args = ap.parse_args()
ROOT = "/home/yasin/Workspace/humanoid_rl"
if args.pol in ("yuru6", "yuru7", "yuru8", "yuru9", "yuru10", "yuru11", "yuru12"):
    from envs.yuru6_env import Yuru6Env as E      # 53-D (phase)
else:
    from envs.yuru4_env import Yuru4Env as E      # 51-D
env = E()
model = PPO.load(f"{ROOT}/models/policy_{args.pol}.zip")
vn = VecNormalize.load(f"{ROOT}/models/vecnorm_{args.pol}.pkl", DummyVecEnv([lambda: E()])); vn.training = False

print(f"LIVE: {args.pol} -- opening window... (to close, close the window)")
obs, _ = env.reset(seed=0)
with mujoco.viewer.launch_passive(env.model, env.data) as v:
    # aim the camera at the robot
    v.cam.distance, v.cam.azimuth, v.cam.elevation = 2.2, 120, -10
    while v.is_running():
        t0 = time.time()
        act, _ = model.predict(vn.normalize_obs(obs), deterministic=True)
        obs, r, te, tr, _ = env.step(act)
        v.cam.lookat[:] = [float(env.data.qpos[0]), 0, 0.3]   # let the camera follow the robot
        v.sync()
        if te or tr:
            obs, _ = env.reset()
        dt = 0.02 - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)                                     # real-time speed (50 Hz)
