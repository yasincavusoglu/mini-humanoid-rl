"""
07_llm_komut_demo.py — LLM HIGH-LEVEL + RL LOW-LEVEL integration.

Turkish natural-language command -> plan (cognitive.llm_planner) -> ACTUALLY run the policies -> video.
  walk_forward = walking policy (up to target distance; with VecNormalize)
  stop         = balance policy (raw obs; robot should stop/stay balanced)
Usage: python scripts/07_llm_komut_demo.py "iki metre ileri yuru sonra dur"
"""
import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch, torch._dynamo, torch.optim         # BEFORE mujoco (segfault fix)
import sys; sys.path.insert(0, "/home/yasin/Workspace/humanoid_rl")
import mujoco
import imageio.v2 as imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.yuruyus_env import YuruyusEnv
from cognitive.llm_planner import komuttan_plan

ROOT = "/home/yasin/Workspace/humanoid_rl"
KOMUT = sys.argv[1] if len(sys.argv) > 1 else "iki metre ileri yuru sonra dur"

# --- World (single env; both walking and balance policies run here) ---
env = YuruyusEnv(render_mode="rgb_array")

# --- Policies ---
pol_yuru = PPO.load(f"{ROOT}/models/policy_yuruyus_dr.zip")   # DR: more stable, never falls
vn_yuru = VecNormalize.load(f"{ROOT}/models/vecnorm_yuruyus_dr.pkl", DummyVecEnv([lambda: YuruyusEnv()]))
vn_yuru.training = False
pol_dur = PPO.load(f"{ROOT}/models/policy_denge.zip")   # balance: trained without VecNormalize -> raw obs

# --- LLM: command -> plan ---
plan = komuttan_plan(KOMUT, ollama_kullan=False)
print("COMMAND :", KOMUT)
print("PLAN    :", plan)

renderer = mujoco.Renderer(env.model, 640, 480)
cam = mujoco.MjvCamera()
cam.distance, cam.azimuth, cam.elevation = 2.2, 90, -8
frames = []
obs, _ = env.reset(seed=3)


def kaydet():
    x = float(env.data.qpos[0])
    cam.lookat[:] = [x, 0, 0.25]
    renderer.update_scene(env.data, cam)
    frames.append(renderer.render())


for adim in plan:
    skill = adim.get("skill")
    if skill == "walk_forward":
        hedef = float(env.data.qpos[0]) + float(adim.get("mesafe", 2.0))
        for _ in range(1500):
            o = vn_yuru.normalize_obs(obs)
            a, _ = pol_yuru.predict(o, deterministic=True)
            obs, r, term, trunc, _ = env.step(a); kaydet()
            if term or float(env.data.qpos[0]) >= hedef:
                break
        print(f"  walk_forward -> x = {float(env.data.qpos[0]):.2f} m")
    elif skill == "stop":
        n = int(float(adim.get("sure", 2.0)) / 0.02)
        for _ in range(max(n, 80)):
            a, _ = pol_dur.predict(obs, deterministic=True)     # denge: ham obs
            obs, r, term, trunc, _ = env.step(a); kaydet()
            if term:
                break
        print(f"  stop -> x = {float(env.data.qpos[0]):.2f} m (stopped)")
    else:
        print(f"  {skill}: policy not available yet (turn/back = requires extra training) -> skipped")

imageio.mimsave(f"{ROOT}/videos/07_llm_komut.mp4", frames, fps=50)
print(f"{len(frames)} frames (~{len(frames)*0.02:.1f} s) -> videos/07_llm_komut.mp4")
