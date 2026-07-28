"""
yuru2_env.py — GAIT-SHAPED walking: REAL WALKING instead of HOPPING.

"Walking characteristics" are added to yuruyus_env's reward (the joint sequence is NOT HARD-CODED;
RL discovers hip-knee-ankle coordination on its own -> the intended gait becomes EMERGENT):
  - both feet in the AIR at the same time  -> HEAVY penalty   (hop/jump killer)
  - exactly ONE foot on the ground (single support) -> reward  (the natural phase of walking)
  - the airborne (swing) foot LIFTS a bit (clearance) -> reward  (not dragging, but STEPPING)
  - torso vertical velocity (bob-bob)  -> penalty
Foot-ground contact is detected via the MuJoCo contact engine (foot geom <-> ground).
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

XML = os.path.join(os.path.dirname(__file__), "..", "models", "mini_humanoid.xml")


class Yuru2Env(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(os.path.abspath(XML))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self._renderer = None
        self._torso = self.model.body("torso").id
        self._zemin = self.model.geom("zemin").id
        self._lfg = self.model.geom("l_foot_g").id
        self._rfg = self.model.geom("r_foot_g").id
        self._lfb = self.model.body("l_foot").id
        self._rfb = self.model.body("r_foot").id

        self.action_space = spaces.Box(-1.0, 1.0, (20,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (51,), np.float32)
        self.ctrl_scale = 0.5
        self.hedef_hiz = 0.4
        self.max_steps = 1000
        self._steps = 0

    def _get_obs(self):
        d = self.data
        return np.concatenate([d.qpos[2:3], d.qpos[3:7], d.qvel[0:6],
                               d.qpos[7:27], d.qvel[6:26]]).astype(np.float32)

    def _diklik(self):
        return float(self.data.xmat[self._torso].reshape(3, 3)[2, 2])

    def _ayak_temas(self):
        sol = sag = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = (c.geom1, c.geom2)
            if self._zemin in pair:
                if self._lfg in pair: sol = True
                if self._rfg in pair: sag = True
        return sol, sag

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[7:27] += self.np_random.uniform(-0.05, 0.05, 20)
        mujoco.mj_forward(self.model, self.data)
        self._steps = 0
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -1, 1) * self.ctrl_scale
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        self._steps += 1

        dik = self._diklik()
        z = float(self.data.qpos[2])
        vx = float(self.data.qvel[0])
        vy = float(self.data.qvel[1])
        vz = float(self.data.qvel[2])
        w = self.data.qvel[3:6]
        sol, sag = self._ayak_temas()
        n = int(sol) + int(sag)
        lfz = float(self.data.xpos[self._lfb][2])
        rfz = float(self.data.xpos[self._rfb][2])

        # --- base (go forward, stay upright/alive) ---
        r_ileri = min(vx, self.hedef_hiz) * 1.5
        r_canli = 0.2
        r_dik = dik * 0.2
        # --- GAIT SHAPING (walking characteristics) ---
        c_ucus = 1.0 if n == 0 else 0.0                       # both feet airborne = hop penalty (most important)
        r_tek = 0.35 if n == 1 else 0.0                        # single support = walking reward
        c_zipla = 0.6 * abs(vz)                                # torso vertical velocity = jumping penalty
        salinim = max(lfz, rfz) if n >= 1 else 0.0             # height of the airborne foot while there is support
        r_clear = min(max(salinim - 0.060, 0.0), 0.05) * 3.0   # foot clearance (stepping, ~2.5cm+)
        # --- penalties ---
        c_enerji = 0.001 * float(np.sum(np.square(action)))
        c_yalpa = 0.02 * float(np.linalg.norm(w))
        c_yana = 0.10 * abs(vy)

        odul = (r_ileri + r_canli + r_dik + r_tek + r_clear
                - c_ucus - c_zipla - c_enerji - c_yalpa - c_yana)

        terminated = (z < 0.25) or (dik < 0.4)
        truncated = self._steps >= self.max_steps
        return self._get_obs(), odul, bool(terminated), bool(truncated), {}

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 480)
        cam = mujoco.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = 2.0, 90, -8
        cam.lookat[:] = [float(self.data.qpos[0]), 0, 0.25]
        self._renderer.update_scene(self.data, cam)
        return self._renderer.render()
