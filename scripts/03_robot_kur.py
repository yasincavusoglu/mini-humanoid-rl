"""
03_robot_kur.py — Lesson 3: LOAD, VALIDATE and SEE our own robot.
1) Loads the MJCF, prints the model statistics + servo torque ceilings.
2) Takes a PNG of the robot standing upright.
3) Records what happens when control=0 (NO balance policy -> topples over).
"""
import os
os.environ["MUJOCO_GL"] = "osmesa"     # GPU broken -> CPU render
import mujoco
import imageio.v2 as imageio

XML = "/home/yasin/Workspace/humanoid_rl/models/mini_humanoid.xml"
OUT = "/home/yasin/Workspace/humanoid_rl/videos"
os.makedirs(OUT, exist_ok=True)

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)

print("=== MODEL LOADED ===")
print(f"  Bodies (body)       : {m.nbody}  (world included)")
print(f"  Deg. of freedom nv  : {m.nv}   (6 free base + 20 joints = 26 expected)")
print(f"  Actuators (servo) nu : {m.nu}")
print(f"  TOTAL MASS          : {m.body_mass.sum():.2f} kg  (target <=4.5)")
print()
print("=== SERVO TORQUE CEILINGS (realism layer) ===")
for i in range(m.nu):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  {name:16s}  +-{m.actuator_forcerange[i][1]:.1f} Nm")

# Camera
cam = mujoco.MjvCamera()
cam.distance, cam.azimuth, cam.elevation = 1.3, 130, -12
cam.lookat[:] = [0, 0, 0.28]
rr = mujoco.Renderer(m, 720, 720)

# --- PNG: robot standing upright (no dynamics, pose only) ---
mujoco.mj_forward(m, d)
rr.update_scene(d, cam)
imageio.imwrite(f"{OUT}/3_robot_ayakta.png", rr.render())
print(f"\nPNG   -> {OUT}/3_robot_ayakta.png  (our robot standing)")

# --- Video: control=0, NO balance -> topples over ---
mujoco.mj_resetData(m, d)
frames = []
for i in range(250):
    d.ctrl[:] = 0.0                    # all servos at neutral angle; but no balance control
    mujoco.mj_step(m, d)
    rr.update_scene(d, cam)
    frames.append(rr.render())
imageio.mimsave(f"{OUT}/4_robot_dengesiz.mp4", frames, fps=50)
print(f"Video -> {OUT}/4_robot_dengesiz.mp4  (no balance policy -> topples over)")
print("\nLESSON: We built the model, but it CANNOT STAND -> no BALANCE POLICY yet.")
print("      Next step (Task-4): teach it to 'stand upright' with RL.")
