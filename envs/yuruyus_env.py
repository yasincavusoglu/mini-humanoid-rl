"""
yuruyus_env.py — the GAME that teaches our robot to WALK FORWARD.

Sibling of denge_env.py (DengeEnv). The skeleton is IDENTICAL:
  reset()  -> start a new game (set up the robot, SMALL noise; NO PUSH)
  step(a)  -> apply one action, return the outcome + REWARD, check if it fell
  obs      -> SAME 51 dims (so the same training script runs)
  reward   -> rewards FORWARD SPEED instead of balance (locomotion best-practice)

DIFF 1: The goal is not to stand still, but to walk FORWARD along the X axis.
DIFF 2: no push on reset -- learning to walk is already very hard;
        if it were pushed on top of that, the agent could learn nothing.
DIFF 3: The reward saturates forward speed (via min) so that the robot
        walks cleanly and under control instead of 'running off / flying'.
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")   # GPU broken -> CPU render
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

XML = os.path.join(os.path.dirname(__file__), "..", "models", "mini_humanoid.xml")


class YuruyusEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(os.path.abspath(XML))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._renderer = None
        self._torso_id = self.model.body("torso").id

        # ACTION: [-1,1] command for 20 servos  (later scaled to the target angle)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(20,), dtype=np.float32)
        # OBSERVATION: SAME 51 dims as DengeEnv -> the same training script runs
        #   height(1) + orientation/IMU(4) + torso velocity(6) + joint angles(20) + joint velocities(20)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(51,), dtype=np.float32)

        self.ctrl_scale = 0.5     # [-1,1] -> +-0.5 rad servo target
        self.hedef_hiz  = 0.4     # m/s: reward increases up to this speed, then saturates
        self.max_steps  = 1000    # longer episode for walking
        self._steps = 0

    def _get_obs(self):
        d = self.data
        return np.concatenate([
            d.qpos[2:3],     # torso height
            d.qpos[3:7],     # torso orientation (quaternion = IMU)
            d.qvel[0:6],     # torso linear + angular velocity
            d.qpos[7:27],    # 20 joint angles
            d.qvel[6:26],    # 20 joint velocities
        ]).astype(np.float32)

    def _diklik(self):
        # alignment of the torso z-axis with world z: 1.0=fully upright, 0=lying flat
        return float(self.data.xmat[self._torso_id].reshape(3, 3)[2, 2])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # ONLY small joint noise -- NO PUSH (walking is already hard).
        self.data.qpos[7:27] += self.np_random.uniform(-0.05, 0.05, size=20)
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        return self._get_obs(), {}

    def step(self, action):
        # 1) convert the action into servo target angles
        self.data.ctrl[:] = np.clip(action, -1, 1) * self.ctrl_scale
        # 2) advance the physics (control ~50 Hz, physics 200 Hz -> 4 sub-steps)
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        self._steps += 1

        # --- measurements ---
        dik   = self._diklik()
        z     = float(self.data.qpos[2])
        vx    = float(self.data.qvel[0])          # forward (X) linear velocity
        vy    = float(self.data.qvel[1])          # lateral (Y) velocity -> undesirable
        wxyz  = self.data.qvel[3:6]               # torso angular velocity -> wobble

        # 3) REWARD (locomotion best-practice):
        #   + forward: forward speed that SATURATES at the target speed (so it walks cleanly, doesn't run off)
        r_ileri  = min(vx, self.hedef_hiz) * 1.5
        #   + alive bonus: constant incentive while it hasn't fallen (0.5->0.2: don't let 'just standing there'
        #     be tempting; lowered to encourage walking -- locomotion local-optimum trap)
        r_canli  = 0.2
        #   + stay upright: the torso's uprightness
        r_dik    = dik * 0.3
        #   - energy: penalize unnecessarily large commands
        c_enerji = 0.001 * float(np.sum(np.square(action)))
        #   - wobble: penalty on torso angular velocity (swaying)
        c_yalpa  = 0.02 * float(np.linalg.norm(wxyz))
        #   - lateral drift: penalize drifting along the Y axis
        c_yana   = 0.1 * abs(vy)

        odul = r_ileri + r_canli + r_dik - c_enerji - c_yalpa - c_yana

        # 4) did it fall? (the game ends if it dropped too low or tilted too far)
        terminated = (z < 0.25) or (dik < 0.4)
        truncated  = self._steps >= self.max_steps
        return self._get_obs(), odul, bool(terminated), bool(truncated), {}

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 480)
        cam = mujoco.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = 1.3, 130, -12
        cam.lookat[:] = [0, 0, 0.28]
        self._renderer.update_scene(self.data, cam)
        return self._renderer.render()
