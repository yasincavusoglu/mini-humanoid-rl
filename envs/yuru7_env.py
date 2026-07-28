"""
yuru7_env.py — yuru6 (phase-clock, weak ankle) + MORE KNEE BEND + TREMOR/LATERAL-SWAY penalty.

Goal: knee should move more + eliminate side-to-side trembling.
Added (on top of yuru6):
  + r_diz  : make the swing leg's KNEE bend more (direct reward)
  - c_roll : keep the torso from tilting SIDEWAYS (roll)  -> cuts lateral swaying
  - c_yalpa: torso angular-velocity penalty (absent in yuru6) -> general tremor
  - c_smooth: penalty on SUDDEN action change      -> cuts high-frequency SHAKE/tremor
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

XML = os.path.join(os.path.dirname(__file__), "..", "models", "mini_humanoid.xml")
DT = 0.02


class Yuru7Env(gym.Env):
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
        self.observation_space = spaces.Box(-np.inf, np.inf, (53,), np.float32)
        self.ctrl_scale = 0.5
        self.hedef_hiz = 0.4
        self.freq = 0.85
        self.max_steps = 1000
        self._steps = 0
        self._phase = 0.0
        self._prev_lf = np.zeros(2); self._prev_rf = np.zeros(2)
        self._prev_a = np.zeros(20)

    def _get_obs(self):
        d = self.data
        base = np.concatenate([d.qpos[2:3], d.qpos[3:7], d.qvel[0:6], d.qpos[7:27], d.qvel[6:26]])
        faz = [np.sin(2 * np.pi * self._phase), np.cos(2 * np.pi * self._phase)]
        return np.concatenate([base, faz]).astype(np.float32)

    def _R(self):
        return self.data.xmat[self._torso].reshape(3, 3)

    def _ayak_temas(self):
        sol = sag = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]; pair = (c.geom1, c.geom2)
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
        self._phase = float(self.np_random.uniform(0.0, 1.0))
        self._prev_lf = self.data.xpos[self._lfb][:2].copy()
        self._prev_rf = self.data.xpos[self._rfb][:2].copy()
        self._prev_a = np.zeros(20)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1)
        self.data.ctrl[:] = action * self.ctrl_scale
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        self._steps += 1
        self._phase = (self._phase + self.freq * DT) % 1.0

        R = self._R()
        dik = float(R[2, 2]); pitch = float(np.arcsin(np.clip(R[0, 2], -1, 1)))
        roll = float(np.arcsin(np.clip(R[1, 2], -1, 1)))          # torso lateral tilt
        z = float(self.data.qpos[2])
        vx = float(self.data.qvel[0]); vy = float(self.data.qvel[1]); vz = float(self.data.qvel[2])
        wvel = self.data.qvel[3:6]
        sol, sag = self._ayak_temas()

        lf_xy = self.data.xpos[self._lfb][:2].copy(); rf_xy = self.data.xpos[self._rfb][:2].copy()
        lf_slip = float(np.linalg.norm(lf_xy - self._prev_lf) / DT) if sol else 0.0
        rf_slip = float(np.linalg.norm(rf_xy - self._prev_rf) / DT) if sag else 0.0
        self._prev_lf, self._prev_rf = lf_xy, rf_xy
        lfz = float(self.data.xpos[self._lfb][2]); rfz = float(self.data.xpos[self._rfb][2])
        lk = float(self.data.qpos[10]); rk = float(self.data.qpos[16])

        sol_stance = self._phase < 0.5
        sag_stance = not sol_stance
        r_faz = 0.5 * float(sol == sol_stance) + 0.5 * float(sag == sag_stance)
        r_clear = 0.0
        if not sol_stance: r_clear += min(max(lfz - 0.06, 0.0), 0.06)
        if not sag_stance: r_clear += min(max(rfz - 0.06, 0.0), 0.06)
        r_clear *= 3.0
        # MORE KNEE BEND: make the swing leg's knee bend (reward up to 0.8 rad ~ 46 degrees)
        r_diz = 0.0
        if not sol_stance: r_diz += min(max(lk, 0.0), 0.8)
        if not sag_stance: r_diz += min(max(rk, 0.0), 0.8)
        r_diz *= 0.6

        r_ileri = min(vx, self.hedef_hiz) * 1.2
        r_canli = 0.1
        r_dikz = dik * 0.1

        # ANTI-TREMOR / LATERAL-SWAY penalties
        c_kayma = 0.6 * (lf_slip + rf_slip)
        c_lean = 1.0 * abs(pitch)
        c_roll = 1.5 * abs(roll)                                  # keep the torso from tilting sideways
        c_yalpa = 0.06 * float(np.linalg.norm(wvel))              # angular velocity (general tremor)
        c_zipla = 0.4 * abs(vz)
        c_yana = 0.20 * abs(vy)
        c_enerji = 0.0005 * float(np.sum(np.square(action)))
        c_smooth = 0.08 * float(np.sum(np.abs(action - self._prev_a)))   # sudden action change = tremor
        self._prev_a = action.copy()
        # ankle-cheat penalty
        qv = self.data.qvel
        c_bilek = 0.05 * (max(0.0, abs(qv[10]) + abs(qv[11]) - abs(qv[8]) - abs(qv[9]))
                          + max(0.0, abs(qv[16]) + abs(qv[17]) - abs(qv[14]) - abs(qv[15])))

        odul = (r_faz + r_clear + r_diz + r_ileri + r_canli + r_dikz
                - c_kayma - c_lean - c_roll - c_yalpa - c_zipla - c_yana - c_enerji - c_smooth - c_bilek)

        terminated = (z < 0.25) or (dik < 0.5)
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
