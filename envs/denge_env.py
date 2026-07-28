"""
denge_env.py — The GAME that teaches our robot to STAND UPRIGHT (keep its balance when pushed).

A 'Gym environment' = the rules of the game. It has 4 parts:
  reset()  -> start a new game (set up the robot + push it randomly)
  step(a)  -> apply one move, return the result + REWARD, check if it fell
  observation -> the numbers the player (agent) sees (IMU + joints)
  reward   -> the score of how well it is doing (stay upright + alive = plus)
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")   # GPU broken -> CPU render
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

XML = os.path.join(os.path.dirname(__file__), "..", "models", "mini_humanoid.xml")


class DengeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(os.path.abspath(XML))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._renderer = None
        self._torso_id = self.model.body("torso").id

        # ACTION: [-1,1] command for 20 servos  (later scaled to a target angle)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(20,), dtype=np.float32)
        # OBSERVATION: height(1) + orientation/IMU(4) + torso vel(6) + joint angle(20) + joint vel(20) = 51
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(51,), dtype=np.float32)

        self.ctrl_scale = 0.5     # [-1,1] -> +-0.5 rad servo target
        self.itme_gucu  = 0.4     # initial 'shove' strength (EXPERIMENT: increase/decrease)
        self.max_steps  = 500
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
        # alignment of the torso z-axis with the world z: 1.0=fully upright, 0=lying down
        return float(self.data.xmat[self._torso_id].reshape(3, 3)[2, 2])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        # Small random perturbation + random PUSH to the torso (that's why it must learn)
        self.data.qpos[7:27] += self.np_random.uniform(-0.05, 0.05, size=20)
        self.data.qvel[0:6]  += self.np_random.uniform(-self.itme_gucu, self.itme_gucu, size=6)
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        return self._get_obs(), {}

    def step(self, action):
        # 1) convert the action to a servo target angle
        self.data.ctrl[:] = np.clip(action, -1, 1) * self.ctrl_scale
        # 2) advance the physics (control ~50 Hz, physics 200 Hz -> 4 sub-steps)
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        self._steps += 1

        # 3) REWARD (heart of the game): stay alive(+1) + stand upright(+uprightness) - energy penalty
        dik = self._diklik()
        z = float(self.data.qpos[2])
        odul = 1.0 + dik - 0.001 * float(np.sum(np.square(action)))

        # 4) did it fall? (game ends if it dropped too low or leaned too far)
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
