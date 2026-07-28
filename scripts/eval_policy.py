"""
eval_policy.py — Run a trained policy + capture VIDEO.
Usage: python scripts/eval_policy.py --env denge
"""
import os, sys, argparse
sys.path.insert(0, "/home/yasin/Workspace/humanoid_rl")
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # don't touch the broken NVIDIA driver -> torch pure CPU
# CRITICAL: load torch._dynamo BEFORE mujoco (otherwise mujoco+torch segfaults)
import torch          # noqa: E402
import torch._dynamo  # noqa: E402
import torch.optim    # noqa: E402
import imageio.v2 as imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def make_env(name):
    if name == "denge":
        from envs.denge_env import DengeEnv
        return DengeEnv(render_mode="rgb_array")
    if name == "yuruyus":
        from envs.yuruyus_env import YuruyusEnv
        return YuruyusEnv(render_mode="rgb_array")
    if name == "yuruyus_dr":
        from envs.yuruyus_dr_env import YuruyusDREnv
        return YuruyusDREnv(render_mode="rgb_array")
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["denge", "yuruyus", "yuruyus_dr"])
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()
    ROOT = "/home/yasin/Workspace/humanoid_rl"

    env = make_env(args.env)
    model = PPO.load(f"{ROOT}/models/policy_{args.env}.zip")
    # If VecNormalize was used during training, we must normalize the observation with the same statistics
    vn_path = f"{ROOT}/models/vecnorm_{args.env}.pkl"
    vec_norm = None
    if os.path.exists(vn_path):
        vec_norm = VecNormalize.load(vn_path, DummyVecEnv([lambda: make_env(args.env)]))
        vec_norm.training = False
        print("  (VecNormalize statistics loaded)")
    obs, _ = env.reset(seed=0)
    frames, toplam, bolum = [], 0.0, 1
    for i in range(args.steps):
        o = vec_norm.normalize_obs(obs) if vec_norm is not None else obs
        a, _ = model.predict(o, deterministic=True)
        obs, r, term, trunc, _ = env.step(a)
        toplam += r
        frames.append(env.render())
        if term or trunc:
            print(f"  episode {bolum}: {'FELL' if term else 'STANDING(time up)'} @step {i+1} | reward {toplam:.1f}")
            obs, _ = env.reset(); toplam = 0.0; bolum += 1
    out = f"{ROOT}/videos/egitilmis_{args.env}.mp4"
    imageio.mimsave(out, frames, fps=50)
    print("Video ->", out)


if __name__ == "__main__":
    main()
