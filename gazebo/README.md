# mini_humanoid — ROS2 / Gazebo bridge

The files needed to run the balance policy we trained in MuJoCo
(`models/policy_denge.zip`) on the ROS2 + Gazebo side. Goal: **a sim-to-real bridge** —
describe the same robot in two languages (MJCF and URDF), then try the trained brain in a
simulator that is closer to reality.

## Files

| File | What it does |
|-------|-------------|
| `mini_humanoid.urdf` | The ROS/Gazebo counterpart of the MJCF. 20 joints (same names/axes/limits), a ros2_control interface, a torso IMU sensor. |
| `mini_humanoid.launch.py` | Start Gazebo + spawn the robot + load the controllers (skeleton). |
| `../ros2/policy_runner.py` | rclpy node running the PPO policy at 50 Hz: `/joint_states`+`/imu` -> 51-dim observation -> 20 actions -> command. |
| `../ros2/engel_kacis_stub.py` | Simple safety reflex: `/scan` distance < threshold -> stop/turn. |

## URDF vs MJCF difference (important)

In MJCF **multiple joints** can be attached to a single body (hip = 3 joints at the same
point). In URDF each joint connects **exactly 2 links** and carries a **single** degree of
freedom. That is why we chained the multi-DOF joints with **massless dummy intermediate
links** (`l_hip_yaw_link`, `l_hip_roll_link`, `l_ankle_pitch_link`, ...). The joint **names,
axes and limits** are identical to the MJCF (MJCF degrees -> URDF radians).

Verified: `check_urdf mini_humanoid.urdf` -> full kinematic tree, 20 revolute joints.

## Observation order (critical)

`policy_runner` builds the observation vector **exactly the same** as
`DengeEnv._get_obs()`, otherwise the policy goes haywire:

```
[ torso_z(1), torso_quat wxyz(4), torso_vel 6dof(6),
  20 joint angles (actuator order), 20 joint velocities ]  = 51
```

**Honesty note:** the Gazebo IMU gives orientation + angular velocity; but the torso
**height** and **linear velocity** do not come directly. In reality these come from
odometry / TF / a state estimator. The code fills these fields with 0/constant for now and
leaves a `TODO`. For walking stability this estimator needs to be added.

## Setup / running

**ROS2 Humble is installed** on this machine (`/opt/ros/humble`). Gazebo Classic +
ros2_control packages are required; if missing:

```bash
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control \
                 ros-humble-ros2-control ros-humble-ros2-controllers
```

For real running the files must be placed into a ROS2 package (`mini_humanoid_bringup`):
`urdf/`, `config/controllers.yaml`, `launch/`. Skeleton flow:

```bash
source /opt/ros/humble/setup.bash
# 1) (once) create the package, copy the files, colcon build, source install/setup.bash
# 2) start the simulation:
ros2 launch mini_humanoid_bringup mini_humanoid.launch.py
# 3) run the policy in a separate terminal:
ros2 run mini_humanoid_bringup policy_runner        # or: python3 ros2/policy_runner.py
```

`controllers.yaml` skeleton (example):

```yaml
controller_manager:
  ros__parameters:
    update_rate: 100
    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster
    position_controller:
      type: position_controllers/JointGroupPositionController

position_controller:
  ros__parameters:
    joints:
      - l_hip_yaw
      - l_hip_roll
      - l_hip_pitch
      - l_knee
      - l_ankle_pitch
      - l_ankle_roll
      - r_hip_yaw
      - r_hip_roll
      - r_hip_pitch
      - r_knee
      - r_ankle_pitch
      - r_ankle_roll
      - l_shoulder_pitch
      - l_shoulder_roll
      - l_elbow
      - r_shoulder_pitch
      - r_shoulder_roll
      - r_elbow
      - neck_yaw
      - neck_pitch
```

> The joint order in `position_controller` must be the **same** as `policy_runner.EKLEM_SIRASI`;
> the 20 values in the `/position_controller/commands` message are mapped in this order.

## Extra notes

- **GPU:** the GPU drivers are broken on this machine; MuJoCo training runs on CPU. Gazebo's
  3D render may need a GPU -> a **reboot / driver fix** may be required.
- **sim-to-real gaps:** (1) the height/linear-velocity estimator is missing (see above);
  (2) the MJCF position-servo `kp` and the Gazebo controller gain are not the same -> tuning
  needed; (3) MuJoCo and Gazebo contact/friction physics differ -> domain randomization is
  recommended; (4) mass/inertia were computed approximately in the URDF, should be updated
  with real CAD.
- **Obstacle avoidance** (`engel_kacis_stub.py`) is a safety layer running **on top of** the
  walking policy; in reality it is wired to override the `policy_runner` output.
