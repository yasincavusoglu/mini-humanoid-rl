"""
13_gait_tam.py — FULL gait analysis: all metrics + LEAD-LEG alternation + FALL count + video.
Usage: python scripts/13_gait_tam.py --pol yuru4   (yuru3/yuru2 also work; same dynamics)
"""
import os, argparse
os.environ["MUJOCO_GL"] = "osmesa"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch, torch._dynamo, torch.optim
import sys; sys.path.insert(0, "/home/yasin/Workspace/humanoid_rl")
import numpy as np, mujoco, imageio.v2 as imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
ap = argparse.ArgumentParser(); ap.add_argument("--pol", default="yuru6"); args = ap.parse_args()
ROOT = "/home/yasin/Workspace/humanoid_rl"
if args.pol in ("yuru6", "yuru7", "yuru8", "yuru9", "yuru10", "yuru11", "yuru12"):
    from envs.yuru6_env import Yuru6Env as EnvCls   # 53-D obs (phase clock); yuru7-12 same obs
else:
    from envs.yuru4_env import Yuru4Env as EnvCls   # 51-D (yuru2-5, same dynamics)
env = EnvCls(render_mode="rgb_array")
model = PPO.load(f"{ROOT}/models/policy_{args.pol}.zip")
vn = VecNormalize.load(f"{ROOT}/models/vecnorm_{args.pol}.pkl", DummyVecEnv([lambda: EnvCls()])); vn.training = False
lfb, rfb = env._lfb, env._rfb

renderer = mujoco.Renderer(env.model, 640, 480)
cam = mujoco.MjvCamera(); cam.distance, cam.azimuth, cam.elevation = 2.0, 90, -8
obs, _ = env.reset(seed=5)
lz, rz, pitch, roll, slip, lk, rk, lhp, rhp, vxs, leads, yanal, frames = [], [], [], [], [], [], [], [], [], [], [], [], []
prev_lf = env.data.xpos[lfb][:2].copy(); prev_rf = env.data.xpos[rfb][:2].copy()
dusme = 0; adim = 0
for i in range(600):
    a, _ = model.predict(vn.normalize_obs(obs), deterministic=True)
    obs, r, te, tr, _ = env.step(a); adim += 1
    sol, sag = env._ayak_temas()
    R = env.data.xmat[env._torso].reshape(3, 3)
    lf = env.data.xpos[lfb][:2].copy(); rf = env.data.xpos[rfb][:2].copy()
    if sol: slip.append(np.linalg.norm(lf - prev_lf) / 0.02)
    if sag: slip.append(np.linalg.norm(rf - prev_rf) / 0.02)
    prev_lf, prev_rf = lf, rf
    lz.append(env.data.xpos[lfb][2]); rz.append(env.data.xpos[rfb][2])
    pitch.append(np.degrees(np.arcsin(np.clip(R[0, 2], -1, 1))))
    roll.append(np.degrees(np.arcsin(np.clip(R[1, 2], -1, 1))))
    lk.append(np.degrees(env.data.qpos[10])); rk.append(np.degrees(env.data.qpos[16]))
    lhp.append(np.degrees(env.data.qpos[9])); rhp.append(np.degrees(env.data.qpos[15]))
    vxs.append(env.data.qvel[0])
    dx = float(env.data.xpos[lfb][0] - env.data.xpos[rfb][0])
    leads.append(1 if dx > 0.02 else (-1 if dx < -0.02 else 0))
    yanal.append(abs(float(lf[1])) + abs(float(rf[1])))
    cam.lookat[:] = [float(env.data.qpos[0]), 0, 0.25]; renderer.update_scene(env.data, cam); frames.append(renderer.render())
    if te or tr:
        dusme += 1
        obs, _ = env.reset()
        prev_lf = env.data.xpos[lfb][:2].copy(); prev_rf = env.data.xpos[rfb][:2].copy()

lz, rz = np.array(lz), np.array(rz); leads = np.array(leads)
kor = float(np.corrcoef(lz - lz.mean(), rz - rz.mean())[0, 1]); esik = min(lz.min(), rz.min()) + 0.025
flight = float(np.mean((lz > esik) & (rz > esik)))
nz = leads[leads != 0]; sw = int(np.sum(np.diff(nz) != 0)) if len(nz) > 1 else 0
imageio.mimsave(f"{ROOT}/videos/gait_{args.pol}.mp4", frames, fps=50)
idx = np.linspace(0, len(frames) - 1, 10).astype(int)
imageio.imwrite(f"{ROOT}/videos/gait_{args.pol}_dense.png", np.concatenate([frames[i] for i in idx], axis=1))
print(f"=== {args.pol} GAIT ({adim} steps) ===")
print(f"FALLS           : {dusme} times (0=never fell, good)")
print(f"LEAD LEG        : left {100*np.mean(leads==1):.0f}%  right {100*np.mean(leads==-1):.0f}%  | switches {sw} times  (both ~50% + many switches = ALTERNATION, good)")
print(f"forward speed   : {np.mean(vxs):+.2f} m/s")
print(f"foot alternation: corr {kor:+.2f} (<0 good) | both-feet-airborne {flight:.2f}")
print(f"STANCE FOOT SLIP : {np.mean(slip) if slip else 0:.2f} m/s")
print(f"torso tilt      : {np.mean(pitch):+.0f} deg (0=upright)")
print(f"SIDE-SWAY       : roll std {np.std(roll):.1f} deg (low=calm, JITTER measure)")
print(f"LATERAL FOOT OUT: {np.mean(yanal)*100:.1f} cm (low=narrow stance; feet should not splay out)")
print(f"knee range      : left {np.ptp(lk):.0f} right {np.ptp(rk):.0f} deg | hip: left {np.ptp(lhp):.0f} right {np.ptp(rhp):.0f} deg")
print(f"-> videos/gait_{args.pol}.mp4 + gait_{args.pol}_dense.png")
