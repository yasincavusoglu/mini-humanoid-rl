"""
10_kol_demo.py — ARM USE: while the robot stands upright in a neutral pose, it SALUTES/WAVES with its RIGHT ARM.

Target angles (radians) are sent DIRECTLY to the position actuators (no +-0.5 limit of env.step),
so the arm rises fully. Legs/torso neutral (0) -> position servos hold balance.
Actuator order: 15=r_sh_pitch, 16=r_sh_roll, 17=r_elbow.
"""
import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import mujoco
import imageio.v2 as imageio
import sys; sys.path.insert(0, "/home/yasin/Workspace/humanoid_rl")
from envs.yuruyus_env import YuruyusEnv

ROOT = "/home/yasin/Workspace/humanoid_rl"
env = YuruyusEnv(render_mode="rgb_array")
renderer = mujoco.Renderer(env.model, 640, 480)
cam = mujoco.MjvCamera()
cam.distance, cam.azimuth, cam.elevation = 1.7, 35, -5
cam.lookat[:] = [0, 0, 0.33]

env.reset(seed=1)
frames = []
for i in range(340):
    t = i * 0.02
    ctrl = np.zeros(20)                              # all joints neutral (balance)
    ctrl[16] = -1.35                                 # right shoulder roll: raise arm OUT-AND-UP
    ctrl[15] = -0.35                                 # right shoulder pitch: slightly forward
    ctrl[17] = -1.0 + 0.75 * np.sin(2 * np.pi * 1.8 * t)   # right elbow: WAVE
    env.data.ctrl[:] = ctrl
    for _ in range(4):
        mujoco.mj_step(env.model, env.data)
    cam.lookat[:] = [float(env.data.qpos[0]), 0, 0.33]
    renderer.update_scene(env.data, cam)
    frames.append(renderer.render())

imageio.mimsave(f"{ROOT}/videos/10_kol_selam.mp4", frames, fps=50)
imageio.imwrite(f"{ROOT}/videos/10_kol_kare.png", frames[220])
print(f"{len(frames)} frames (~{len(frames)*0.02:.1f} s) -> videos/10_kol_selam.mp4")
