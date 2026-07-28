"""
yuru6_env.py — PHASE-BASED GAIT CLOCK (definitively fixes the limping).

Problem: the manual reward variants (yuru2-5) always converged to a STABLE BUT LIMPING gait (one leg leading).
Solution (the real bipedal RL standard, Siekmann/Cassie): give the policy a periodic CLOCK
signal (phase) + reward each foot's SCHEDULED contact relative to this clock:
  phase [0.0, 0.5)  ->  LEFT down (stance)  + RIGHT lift (swing)
  phase [0.5, 1.0)  ->  RIGHT down (stance)  + LEFT lift (swing)
This way alternation is enforced as a RHYTHM; the robot follows the clock and steps in turn.

The OBSERVATION becomes 53-D (51 + [sin(2*pi*phase), cos(2*pi*phase)]).
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

XML = os.path.join(os.path.dirname(__file__), "..", "models", "mini_humanoid.xml")
DT = 0.02


class Yuru6Env(gym.Env):
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
        self.observation_space = spaces.Box(-np.inf, np.inf, (53,), np.float32)   # 51 + phase(2)
        self.ctrl_scale = 0.5
        self.hedef_hiz = 0.4
        self.freq = 0.85            # gait Hz (full cycle ~1.18s ~ 59 steps)
        self.max_steps = 1000
        self._steps = 0
        self._phase = 0.0
        self._prev_lf = np.zeros(2); self._prev_rf = np.zeros(2)

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
        self._phase = float(self.np_random.uniform(0.0, 1.0))     # random phase start
        self._prev_lf = self.data.xpos[self._lfb][:2].copy()
        self._prev_rf = self.data.xpos[self._rfb][:2].copy()
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -1, 1) * self.ctrl_scale
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        self._steps += 1
        self._phase = (self._phase + self.freq * DT) % 1.0

        R = self._R(); dik = float(R[2, 2]); pitch = float(np.arcsin(np.clip(R[0, 2], -1, 1)))
        z = float(self.data.qpos[2])
        vx = float(self.data.qvel[0]); vy = float(self.data.qvel[1]); vz = float(self.data.qvel[2])
        sol, sag = self._ayak_temas()

        lf_xy = self.data.xpos[self._lfb][:2].copy(); rf_xy = self.data.xpos[self._rfb][:2].copy()
        lf_slip = float(np.linalg.norm(lf_xy - self._prev_lf) / DT) if sol else 0.0
        rf_slip = float(np.linalg.norm(rf_xy - self._prev_rf) / DT) if sag else 0.0
        self._prev_lf, self._prev_rf = lf_xy, rf_xy
        lfz = float(self.data.xpos[self._lfb][2]); rfz = float(self.data.xpos[self._rfb][2])

        # --- PHASE SCHEDULE: which foot should be stance/swing ---
        sol_stance_hedef = self._phase < 0.5
        sag_stance_hedef = not sol_stance_hedef
        # does the contact match the target (MAIN REWARD: enforces alternation as a rhythm)
        r_faz = 0.5 * float(sol == sol_stance_hedef) + 0.5 * float(sag == sag_stance_hedef)
        # clearance of the swing foot (the one that should be lifted)
        r_clear = 0.0
        if not sol_stance_hedef: r_clear += min(max(lfz - 0.06, 0.0), 0.06)
        if not sag_stance_hedef: r_clear += min(max(rfz - 0.06, 0.0), 0.06)
        r_clear *= 3.0

        # --- support terms ---
        r_ileri = min(vx, self.hedef_hiz) * 1.2
        r_canli = 0.1
        r_dikz = dik * 0.1
        c_kayma = 0.6 * (lf_slip + rf_slip)
        c_lean = 1.0 * abs(pitch)
        c_zipla = 0.4 * abs(vz)
        c_yana = 0.15 * abs(vy)
        c_enerji = 0.0005 * float(np.sum(np.square(action)))
        # ANKLE-CHEAT penalty: if the ankle moves but hip+knee do not -> penalty
        qv = self.data.qvel
        lbil = abs(qv[10]) + abs(qv[11]); rbil = abs(qv[16]) + abs(qv[17])   # left/right ankle speed
        lhd = abs(qv[8]) + abs(qv[9]);   rhd = abs(qv[14]) + abs(qv[15])     # left/right hip_pitch+knee speed
        c_bilek = 0.05 * (max(0.0, lbil - lhd) + max(0.0, rbil - rhd))

        odul = (r_faz + r_clear + r_ileri + r_canli + r_dikz
                - c_kayma - c_lean - c_zipla - c_yana - c_enerji - c_bilek)

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
