"""
yuruyus_dr_env.py — a ROBUST walking environment with Domain Randomization (DR).

THE ESSENCE OF THE SIM-TO-REAL BRIDGE: on every trial the robot is spawned with DIFFERENT physics
(mass ±20%, friction ±30%, servo torque ±15%, sensor noise). Because of this the
policy does not latch onto a single physics -> it stays robust on the real robot / in Gazebo (different physics) too.
It learns 'a bit heavier/lighter/slipperier doesn't matter, I still walk'.

REWARD/OBSERVATION are byte-for-byte identical to YuruyusEnv (the same train_ppo.py --env yuruyus_dr runs).
The only difference: on reset() the physics parameters are randomized + noise on observation/action.
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

XML = os.path.join(os.path.dirname(__file__), "..", "models", "mini_humanoid.xml")


class YuruyusDREnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(os.path.abspath(XML))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._renderer = None
        self._torso_id = self.model.body("torso").id

        # store the NOMINAL (real) values -- deviate randomly from these on each reset
        self._m0  = self.model.body_mass.copy()
        self._i0  = self.model.body_inertia.copy()
        self._f0  = self.model.geom_friction.copy()
        self._fr0 = self.model.actuator_forcerange.copy()

        self.action_space = spaces.Box(-1.0, 1.0, shape=(20,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(51,), dtype=np.float32)
        self.ctrl_scale = 0.5
        self.hedef_hiz = 0.4
        self.max_steps = 1000
        self._steps = 0
        self.obs_gurultu = 0.01     # sensor noise
        self.act_gurultu = 0.02     # actuator noise

    def _get_obs(self):
        d = self.data
        o = np.concatenate([d.qpos[2:3], d.qpos[3:7], d.qvel[0:6],
                            d.qpos[7:27], d.qvel[6:26]]).astype(np.float32)
        return o + self.np_random.normal(0, self.obs_gurultu, size=o.shape).astype(np.float32)

    def _diklik(self):
        return float(self.data.xmat[self._torso_id].reshape(3, 3)[2, 2])

    def _randomize_fizik(self):
        u = self.np_random.uniform
        ms = u(0.8, 1.2, size=self._m0.shape)                    # mass ±20%
        self.model.body_mass[:]    = self._m0 * ms
        self.model.body_inertia[:] = self._i0 * ms[:, None]      # keep inertia consistent too
        self.model.geom_friction[:] = self._f0 * u(0.7, 1.3, size=self._f0.shape[0])[:, None]   # friction ±30%
        self.model.actuator_forcerange[:] = self._fr0 * u(0.85, 1.15, size=self._fr0.shape[0])[:, None]  # servo torque ±15%

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._randomize_fizik()
        self.data.qpos[7:27] += self.np_random.uniform(-0.05, 0.05, size=20)
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1) + self.np_random.normal(0, self.act_gurultu, size=20)
        self.data.ctrl[:] = np.clip(action, -1, 1) * self.ctrl_scale
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        self._steps += 1

        dik = self._diklik()
        z   = float(self.data.qpos[2])
        vx  = float(self.data.qvel[0])
        vy  = float(self.data.qvel[1])
        w   = self.data.qvel[3:6]
        odul = (min(vx, self.hedef_hiz) * 1.5 + 0.2 + dik * 0.3
                - 0.001 * float(np.sum(np.square(action)))
                - 0.02 * float(np.linalg.norm(w))
                - 0.1 * abs(vy))
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
