"""
train_ppo.py — TRAIN an environment with PPO (denge or yuruyus).
Usage: python scripts/train_ppo.py --env denge --steps 800000
GPU broken -> CPU (SB3 MlpPolicy is already fast on CPU).
"""
import os, sys, argparse
sys.path.insert(0, "/home/yasin/Workspace/humanoid_rl")
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # don't touch the broken NVIDIA driver -> torch pure CPU
# CRITICAL: load torch._dynamo BEFORE mujoco (otherwise mujoco+torch segfaults)
import torch          # noqa: E402
import torch._dynamo  # noqa: E402
import torch.optim    # noqa: E402
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback


class PeriyodikKaydet(BaseCallback):
    """Save policy + vecnorm every `freq` steps -> resilience to session interruption."""
    def __init__(self, env_name, freq, venv):
        super().__init__()
        self.env_name = env_name; self.freq = freq; self.venv = venv; self._last = 0
    def _on_step(self):
        if self.num_timesteps - self._last >= self.freq:
            self._last = self.num_timesteps
            ROOT = "/home/yasin/Workspace/humanoid_rl"
            self.model.save(f"{ROOT}/models/policy_{self.env_name}.zip")
            self.venv.save(f"{ROOT}/models/vecnorm_{self.env_name}.pkl")
            print(f"[checkpoint] {self.num_timesteps} steps -> policy_{self.env_name}.zip", flush=True)
        return True


def make_env(name, render_mode=None):
    if name == "denge":
        from envs.denge_env import DengeEnv
        return DengeEnv(render_mode=render_mode)
    if name == "yuruyus":
        from envs.yuruyus_env import YuruyusEnv
        return YuruyusEnv(render_mode=render_mode)
    if name == "yuruyus_dr":
        from envs.yuruyus_dr_env import YuruyusDREnv
        return YuruyusDREnv(render_mode=render_mode)
    if name == "yuru2":
        from envs.yuru2_env import Yuru2Env
        return Yuru2Env(render_mode=render_mode)
    if name == "yuru3":
        from envs.yuru3_env import Yuru3Env
        return Yuru3Env(render_mode=render_mode)
    if name == "yuru4":
        from envs.yuru4_env import Yuru4Env
        return Yuru4Env(render_mode=render_mode)
    if name == "yuru5":
        from envs.yuru5_env import Yuru5Env
        return Yuru5Env(render_mode=render_mode)
    if name == "yuru6":
        from envs.yuru6_env import Yuru6Env
        return Yuru6Env(render_mode=render_mode)
    if name == "yuru7":
        from envs.yuru7_env import Yuru7Env
        return Yuru7Env(render_mode=render_mode)
    if name == "yuru8":
        from envs.yuru8_env import Yuru8Env
        return Yuru8Env(render_mode=render_mode)
    if name == "yuru9":
        from envs.yuru9_env import Yuru9Env
        return Yuru9Env(render_mode=render_mode)
    if name == "yuru10":
        from envs.yuru10_env import Yuru10Env
        return Yuru10Env(render_mode=render_mode)
    if name == "yuru11":
        from envs.yuru11_env import Yuru11Env
        return Yuru11Env(render_mode=render_mode)
    if name == "yuru12":
        from envs.yuru12_env import Yuru12Env
        return Yuru12Env(render_mode=render_mode)
    raise ValueError(f"unknown environment: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=["denge", "yuruyus", "yuruyus_dr", "yuru2", "yuru3", "yuru4", "yuru5", "yuru6", "yuru7", "yuru8", "yuru9", "yuru10", "yuru11", "yuru12"])
    ap.add_argument("--steps", type=int, default=800_000)
    ap.add_argument("--n_envs", type=int, default=4)
    ap.add_argument("--subproc", action="store_true")
    args = ap.parse_args()

    ROOT = "/home/yasin/Workspace/humanoid_rl"
    fns = [(lambda: make_env(args.env)) for _ in range(args.n_envs)]
    venv = SubprocVecEnv(fns) if args.subproc else DummyVecEnv(fns)
    venv = VecMonitor(venv)
    # VecNormalize: automatically normalize observation + reward scales -> PPO much more stable
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    model = PPO(
        "MlpPolicy", venv, device="cpu", verbose=1,
        n_steps=2048, batch_size=512, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.0, learning_rate=3e-4,
        target_kl=0.04,   # prevent large/destructive policy jumps (KL explosion fix)
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=f"{ROOT}/logs/{args.env}",
    )
    model.learn(total_timesteps=args.steps, callback=PeriyodikKaydet(args.env, 1_000_000, venv))
    model.save(f"{ROOT}/models/policy_{args.env}.zip")
    venv.save(f"{ROOT}/models/vecnorm_{args.env}.pkl")   # also save the normalization statistics
    print("SAVED:", f"{ROOT}/models/policy_{args.env}.zip")


if __name__ == "__main__":
    main()
