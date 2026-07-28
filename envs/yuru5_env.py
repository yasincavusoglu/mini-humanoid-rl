"""
yuru5_env.py — yuru3 (stable gait) + GENTLE leading-leg alternation.

yuru4 lesson: the leading-leg penalty was too HARSH + UNBOUNDED (0.05*(ayni-55)) -> the robot lost
its balance, fell (5 times), took excessively high steps. yuru3 was stable+alternating but limped.
yuru5 = yuru3's stable reward + GENTLY encouraging leading-leg alternation:
  - bonus when the leader CHANGES (positive, safe incentive)
  - a SMALL + CAPPED penalty if the same leg stays forward for TOO long (not enough to break balance)
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

XML = os.path.join(os.path.dirname(__file__), "..", "models", "mini_humanoid.xml")
DT = 0.02


class Yuru5Env(gym.Env):
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
        self.iLHP, self.iLK = 9, 10
        self.iRHP, self.iRK = 15, 16
        self.action_space = spaces.Box(-1.0, 1.0, (20,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (51,), np.float32)
        self.ctrl_scale = 0.5
        self.hedef_hiz = 0.4
        self.max_steps = 1000
        self._steps = 0
        self._prev_lf = np.zeros(2); self._prev_rf = np.zeros(2)
        self._prev_lead = 1; self._ayni = 0

    def _get_obs(self):
        d = self.data
        return np.concatenate([d.qpos[2:3], d.qpos[3:7], d.qvel[0:6],
                               d.qpos[7:27], d.qvel[6:26]]).astype(np.float32)

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
        self._prev_lf = self.data.xpos[self._lfb][:2].copy()
        self._prev_rf = self.data.xpos[self._rfb][:2].copy()
        self._prev_lead = 1 if self.data.xpos[self._lfb][0] >= self.data.xpos[self._rfb][0] else -1
        self._ayni = 0
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -1, 1) * self.ctrl_scale
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        self._steps += 1

        R = self._R(); dik = float(R[2, 2]); pitch = float(np.arcsin(np.clip(R[0, 2], -1, 1)))
        z = float(self.data.qpos[2])
        vx = float(self.data.qvel[0]); vy = float(self.data.qvel[1]); vz = float(self.data.qvel[2])
        w = self.data.qvel[3:6]
        sol, sag = self._ayak_temas(); n = int(sol) + int(sag)

        lf_xy = self.data.xpos[self._lfb][:2].copy(); rf_xy = self.data.xpos[self._rfb][:2].copy()
        lf_slip = float(np.linalg.norm(lf_xy - self._prev_lf) / DT) if sol else 0.0
        rf_slip = float(np.linalg.norm(rf_xy - self._prev_rf) / DT) if sag else 0.0
        self._prev_lf, self._prev_rf = lf_xy, rf_xy

        lfz = float(self.data.xpos[self._lfb][2]); rfz = float(self.data.xpos[self._rfb][2])
        lk = float(self.data.qpos[self.iLK]); rk = float(self.data.qpos[self.iRK])
        lhp = float(self.data.qpos[self.iLHP]); rhp = float(self.data.qpos[self.iRHP])
        l_sw = not sol; r_sw = not sag

        # ---- GENTLE leading-leg alternation ----
        dx = float(lf_xy[0] - rf_xy[0])
        if dx > 0.04:   lead = 1
        elif dx < -0.04: lead = -1
        else:           lead = self._prev_lead
        r_switch = 0.0
        if lead != self._prev_lead:
            self._ayni = 0; r_switch = 0.5
        else:
            self._ayni += 1
        self._prev_lead = lead
        c_ayni = 0.03 * min(max(0, self._ayni - 60), 40)     # GENTLE + CAPPED (max ~1.2, after 60 steps)

        # ---- yuru3 rewards/penalties ----
        r_ileri = min(vx, self.hedef_hiz) * 1.5
        r_canli = 0.2
        r_dikz = dik * 0.1
        r_tek = 0.3 if n == 1 else 0.0
        r_diz = 0.0
        if l_sw: r_diz += min(max(lk, 0.0), 0.6)
        if r_sw: r_diz += min(max(rk, 0.0), 0.6)
        r_diz *= 0.5
        r_stride = min(abs(lhp - rhp), 0.7) * 0.5
        sal = 0.0
        if l_sw: sal = max(sal, lfz)
        if r_sw: sal = max(sal, rfz)
        r_clear = min(max(sal - 0.06, 0.0), 0.06) * 3.0

        c_kayma = 1.0 * (lf_slip + rf_slip)
        c_lean = 1.5 * abs(pitch)
        c_ucus = 1.0 if n == 0 else 0.0
        c_zipla = 0.5 * abs(vz)
        c_yana = 0.15 * abs(vy)
        c_yalpa = 0.02 * float(np.linalg.norm(w))
        c_enerji = 0.0005 * float(np.sum(np.square(action)))

        odul = (r_ileri + r_canli + r_dikz + r_tek + r_diz + r_stride + r_clear + r_switch
                - c_ayni - c_kayma - c_lean - c_ucus - c_zipla - c_yana - c_yalpa - c_enerji)

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
