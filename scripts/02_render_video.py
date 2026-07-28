"""
02_render_video.py — Lesson 2: What is an 'ACTION'? LET'S SEE.

Same robot, same physics. We produce two videos; the only difference is the command SENT to the joints:
  1) ZERO action    -> the brain gives no command      -> lifeless collapse
  2) RANDOM action  -> the brain gives random commands  -> 'drunk baby' flailing
At the end, we print the actual 'action vector' (17 numbers) from the 1st frame of the random video
so you can match the movement you see in the video with concrete numbers.
"""
import os
os.environ["MUJOCO_GL"] = "osmesa"     # GPU driver broken -> CPU software render (skip the GPU)

import gymnasium as gym
import numpy as np
import imageio.v2 as imageio

OUT = "/home/yasin/Workspace/humanoid_rl/videos"
os.makedirs(OUT, exist_ok=True)


def kaydet(dosya, politika, n_adim, baslik, tohum=0):
    env = gym.make("Humanoid-v5", render_mode="rgb_array")
    obs, _ = env.reset(seed=tohum)
    kareler = []
    ilk_aksiyon = None
    for i in range(n_adim):
        a = politika(env, obs, i)
        if i == 0:
            ilk_aksiyon = a
        obs, r, term, trunc, _ = env.step(a)
        kareler.append(env.render())
        if term or trunc:                 # fell/done -> stand it back up, keep recording
            obs, _ = env.reset()
    env.close()
    imageio.mimsave(dosya, kareler, fps=30)
    print(f"  {baslik}: {len(kareler)} frames -> {dosya}")
    return ilk_aksiyon


# 1) ZERO action
kaydet(
    f"{OUT}/1_sifir_aksiyon.mp4",
    lambda env, obs, i: np.zeros(env.action_space.shape, dtype=np.float32),
    n_adim=150, baslik="ZERO ACTION  (lifeless collapse)",
)

# 2) RANDOM action
ornek = kaydet(
    f"{OUT}/2_rastgele_aksiyon.mp4",
    lambda env, obs, i: env.action_space.sample(),
    n_adim=240, baslik="RANDOM ACTION (drunk baby)",
)

print()
print("=== HERE IS AN 'ACTION VECTOR' (1st frame of the random video) ===")
print("Each of these 17 numbers is a 'turn this much' command to ONE joint. The reason for the flailing in the video:")
np.set_printoptions(precision=2, suppress=True)
print(" ", ornek)
print()
print("LESSON: Same robot + same physics. The only change -> the 17 numbers sent to the joints.")
print("      ZERO   -> lifeless collapse.   RANDOM -> flailing and falling.")
print("      TRAINING -> learning to choose these 17 numbers so it 'stays upright/walks'.")
