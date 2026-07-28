"""
politika_numpy.py — Lightweight module that runs the trained policy TORCH-FREE.

Only numpy is required -> runs cleanly in the ROS2/Gazebo system python (alongside rclpy).
The weights are loaded from the .npz produced by scripts/export_policy_numpy.py.

Usage:
    from politika_numpy import NumpyPolitika
    pol = NumpyPolitika("models/politika_yuruyus_numpy.npz")
    aksiyon = pol.act(gozlem_51)   # -> 20 dims, [-1,1]
"""
import numpy as np


class NumpyPolitika:
    def __init__(self, npz_yolu):
        d = np.load(npz_yolu)
        self.mean = d["mean"]; self.var = d["var"]
        self.eps = float(d["eps"]); self.clip = float(d["clip"])
        self.Wa = d["Wa"]; self.ba = d["ba"]
        n = int(d["n"])
        self.W = [d[f"w{i}"] for i in range(n)]
        self.b = [d[f"b{i}"] for i in range(n)]

    def act(self, obs):
        """51-dim observation -> 20-dim action (deterministic, exactly matching SB3)."""
        obs = np.asarray(obs, dtype=np.float64)
        o = np.clip((obs - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip)
        x = o
        for W, b in zip(self.W, self.b):
            x = np.tanh(W @ x + b)
        return np.clip(self.Wa @ x + self.ba, -1.0, 1.0)


if __name__ == "__main__":
    import sys, os
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pol = NumpyPolitika(f"{ROOT}/models/politika_yuruyus_numpy.npz")
    a = pol.act(np.zeros(51))
    print(f"NumpyPolitika OK (torch-free): zero-observation -> action shape {a.shape}, "
          f"range [{a.min():.2f}, {a.max():.2f}]")
