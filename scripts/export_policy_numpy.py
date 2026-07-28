"""
export_policy_numpy.py — Export a trained PPO policy to PURE NUMPY.

WHY: ROS2/Gazebo uses the system python (rclpy); ours is in an isolated SB3/torch venv.
Running both in the same process is hard. Solution: extract the policy's neural network weights +
VecNormalize statistics into a .npz; the ROS2 node runs it with ONLY numpy
(no torch needed). This way the robot brain runs anywhere.

Usage: python scripts/export_policy_numpy.py --env yuruyus
Output: models/politika_{env}_numpy.npz  +  verification (exact match with SB3?)
"""
import os, sys, argparse
sys.path.insert(0, "/home/yasin/Workspace/humanoid_rl")
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch, torch._dynamo, torch.optim   # before mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from train_ppo import make_env

ROOT = "/home/yasin/Workspace/humanoid_rl"


def numpy_act(obs, W, b, Wa, ba, mean, var, eps, clip):
    o = np.clip((obs - mean) / np.sqrt(var + eps), -clip, clip)
    x = o
    for Wi, bi in zip(W, b):
        x = np.tanh(Wi @ x + bi)
    return np.clip(Wa @ x + ba, -1.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    args = ap.parse_args()

    model = PPO.load(f"{ROOT}/models/policy_{args.env}.zip")
    sd = model.policy.state_dict()
    # collect the Linear layers in the policy net in order
    wkeys = sorted([k for k in sd if k.startswith("mlp_extractor.policy_net") and k.endswith(".weight")],
                   key=lambda k: int(k.split(".")[2]))
    W = [sd[k].cpu().numpy() for k in wkeys]
    b = [sd[k.rsplit(".", 1)[0] + ".bias"].cpu().numpy() for k in wkeys]
    Wa = sd["action_net.weight"].cpu().numpy()
    ba = sd["action_net.bias"].cpu().numpy()

    # VecNormalize statistics
    vn = VecNormalize.load(f"{ROOT}/models/vecnorm_{args.env}.pkl",
                           DummyVecEnv([lambda: make_env(args.env)]))
    mean = vn.obs_rms.mean.astype(np.float64)
    var = vn.obs_rms.var.astype(np.float64)
    eps = float(vn.epsilon)
    clip = float(vn.clip_obs)

    out = f"{ROOT}/models/politika_{args.env}_numpy.npz"
    save = {"mean": mean, "var": var, "eps": eps, "clip": clip, "Wa": Wa, "ba": ba, "n": len(W)}
    for i, (Wi, bi) in enumerate(zip(W, b)):
        save[f"w{i}"] = Wi; save[f"b{i}"] = bi
    np.savez(out, **save)
    print("EXPORT ->", out, f"({len(W)} hidden layers)")

    # VERIFICATION: does the numpy output exactly match SB3?
    env = make_env(args.env)
    obs, _ = env.reset(seed=0)
    farklar = []
    for _ in range(200):
        o_n = vn.normalize_obs(obs)
        a_sb3, _ = model.predict(o_n, deterministic=True)
        a_np = numpy_act(obs.astype(np.float64), W, b, Wa, ba, mean, var, eps, clip)
        farklar.append(float(np.max(np.abs(a_sb3 - a_np))))
        obs, _, term, trunc, _ = env.step(a_sb3)
        if term or trunc:
            obs, _ = env.reset()
    print(f"VERIFICATION: numpy vs SB3 max action difference = {max(farklar):.2e} (should be ≈0)")


if __name__ == "__main__":
    main()
