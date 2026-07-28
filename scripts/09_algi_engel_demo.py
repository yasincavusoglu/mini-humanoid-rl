"""
09_algi_engel_demo.py — PERCEPTION + OBSTACLE AVOIDANCE (stop) demo (MuJoCo).

The robot walks forward with the DR walking policy. A forward-facing DISTANCE SENSOR (rangefinder)
detects an obstacle ahead (< threshold) and the robot switches to the BALANCE policy and STOPS (no collision).
This is the simplest form of the "perception -> decision -> action" chain (walking first, then these).

Note: the base model (mini_humanoid.xml) is UNCHANGED; we build the scene (obstacle + sensor) here
by injecting a string into the XML.
"""
import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch, torch._dynamo, torch.optim              # BEFORE mujoco (segfault fix)
import sys; sys.path.insert(0, "/home/yasin/Workspace/humanoid_rl")
import numpy as np
import mujoco
import imageio.v2 as imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.yuruyus_env import YuruyusEnv

ROOT = "/home/yasin/Workspace/humanoid_rl"

# --- Scene: add a forward-facing sensor + obstacle box to the base model ---
with open(f"{ROOT}/models/mini_humanoid.xml") as f:
    xml = f.read()
xml = xml.replace('<geom name="torso_g"',
                  '<site name="ileri_site" pos="0.06 0 0" zaxis="1 0 0" size="0.008"/>\n      <geom name="torso_g"', 1)
xml = xml.replace('</worldbody>',
                  '  <geom name="engel" type="box" pos="2.6 0 0.25" size="0.08 0.6 0.25" rgba="0.85 0.2 0.2 1"/>\n  </worldbody>', 1)
xml = xml.replace('</mujoco>',
                  '  <sensor><rangefinder name="mesafe" site="ileri_site"/></sensor>\n</mujoco>', 1)

# --- Env (load model with scene) ---
env = YuruyusEnv(render_mode="rgb_array")
env.model = mujoco.MjModel.from_xml_string(xml)
env.data = mujoco.MjData(env.model)
env._torso_id = env.model.body("torso").id
mesafe_adr = env.model.sensor_adr[env.model.sensor("mesafe").id]

# --- Policies: DR walking + balance (stop) ---
pol_yuru = PPO.load(f"{ROOT}/models/policy_yuruyus_dr.zip")
vn = VecNormalize.load(f"{ROOT}/models/vecnorm_yuruyus_dr.pkl", DummyVecEnv([lambda: YuruyusEnv()])); vn.training = False
pol_dur = PPO.load(f"{ROOT}/models/policy_denge.zip")   # balance = raw obs

renderer = mujoco.Renderer(env.model, 640, 480)
cam = mujoco.MjvCamera(); cam.distance, cam.azimuth, cam.elevation = 2.6, 90, -8
obs, _ = env.reset(seed=2)
frames, durum, ESIK = [], "YURUYOR", 0.55
for i in range(800):
    d = float(env.data.sensordata[mesafe_adr])          # forward distance (m); <=0 -> no obstacle
    if 0.0 < d < ESIK and durum == "YURUYOR":
        durum = "DURDU"
        print(f"  OBSTACLE detected! distance={d:.2f} m -> switch to BALANCE (stop), step {i}, x={float(env.data.qpos[0]):.2f} m")
    if durum == "YURUYOR":
        a, _ = pol_yuru.predict(vn.normalize_obs(obs), deterministic=True)
    else:
        a, _ = pol_dur.predict(obs, deterministic=True)
    obs, r, term, trunc, _ = env.step(a)
    cam.lookat[:] = [float(env.data.qpos[0]), 0, 0.25]
    renderer.update_scene(env.data, cam); frames.append(renderer.render())
    if term or (durum == "DURDU" and i > 0 and len(frames) > 250 and float(env.data.qpos[0]) and False):
        pass
    if term:
        break

imageio.mimsave(f"{ROOT}/videos/09_algi_engel.mp4", frames, fps=50)
print(f"{len(frames)} frames (~{len(frames)*0.02:.1f} s) -> videos/09_algi_engel.mp4 | final x={float(env.data.qpos[0]):.2f} m, state={durum} (obstacle x=2.6)")
