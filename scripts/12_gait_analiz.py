"""
12_gait_analiz.py — OBJECTIVELY measure a walking policy's gait (WALKING or HOPPING) + video.
Usage: python scripts/12_gait_analiz.py --pol yuru2   (or yuruyus_dr, yuruyus)

Metrics:
  - left/right foot height correlation: <0 ≈ alternating STEP (walking), >0 ≈ together (hopping)
  - both-feet-airborne-at-once ratio: high ≈ flight/jumping, low ≈ walking
  - torso vertical oscillation: large ≈ bouncing
"""
import os, argparse
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch, torch._dynamo, torch.optim
import sys; sys.path.insert(0, "/home/yasin/Workspace/humanoid_rl")
import numpy as np
import mujoco
import imageio.v2 as imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.yuruyus_env import YuruyusEnv

ap = argparse.ArgumentParser(); ap.add_argument("--pol", default="yuru2"); args = ap.parse_args()
ROOT = "/home/yasin/Workspace/humanoid_rl"

env = YuruyusEnv(render_mode="rgb_array")
model = PPO.load(f"{ROOT}/models/policy_{args.pol}.zip")
vn = VecNormalize.load(f"{ROOT}/models/vecnorm_{args.pol}.pkl", DummyVecEnv([lambda: YuruyusEnv()])); vn.training = False
lf = env.model.body("l_foot").id; rf = env.model.body("r_foot").id

renderer = mujoco.Renderer(env.model, 640, 480)
cam = mujoco.MjvCamera(); cam.distance, cam.azimuth, cam.elevation = 2.0, 90, -8
obs, _ = env.reset(seed=5)
lz, rz, tz, frames = [], [], [], []
for i in range(500):
    a, _ = model.predict(vn.normalize_obs(obs), deterministic=True)
    obs, r, term, trunc, _ = env.step(a)
    lz.append(float(env.data.xpos[lf][2])); rz.append(float(env.data.xpos[rf][2])); tz.append(float(env.data.qpos[2]))
    cam.lookat[:] = [float(env.data.qpos[0]), 0, 0.25]
    renderer.update_scene(env.data, cam); frames.append(renderer.render())
    if term or trunc:
        break

lz, rz, tz = np.array(lz), np.array(rz), np.array(tz)
kor = float(np.corrcoef(lz - lz.mean(), rz - rz.mean())[0, 1])
esik = min(lz.min(), rz.min()) + 0.025
flight = float(np.mean((lz > esik) & (rz > esik)))
salinim = float(tz.std()) * 100
imageio.mimsave(f"{ROOT}/videos/gait_{args.pol}.mp4", frames, fps=50)

print(f"[{args.pol}] forward {float(env.data.qpos[0]):.2f} m | foot-corr {kor:+.2f} | both-feet-airborne {flight:.2f} | torso-sway {salinim:.1f} cm")
print(f"[{args.pol}] VERDICT: {'HOPPING' if (kor > 0.1 or flight > 0.15) else 'WALKING (alternating)'}  -> videos/gait_{args.pol}.mp4")
