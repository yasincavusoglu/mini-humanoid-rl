#!/usr/bin/env python3
"""
policy_runner.py — ROS2 node that runs the trained PPO policy in REAL TIME (SKELETON).

IDEA (sim-to-real bridge):
  The brain we trained in MuJoCo (models/policy_denge.zip) drives the Gazebo/real robot here.
  Loop:    sensors (/joint_states + /imu)  ->  51-dim OBSERVATION  ->  PPO  ->  20 ACTIONS
           ->  joint position commands  ->  robot.

  CRITICAL: We must build the observation vector in the EXACT same order as DengeEnv._get_obs(),
  otherwise the policy sees 'foreign' numbers and produces garbage.  Order:
     [ torso_z(1), torso_quat wxyz(4), torso_vel 6dof(6),
       20 joint angles (actuator order), 20 joint velocities ] = 51

  NOTE (honesty): The Gazebo IMU gives orientation + angular velocity; but body HEIGHT and
  LINEAR VELOCITY don't come directly.  In reality these come from odometry/TF/state-estimator.
  Here we fill the relevant fields with 0 and leave a TODO (see below).

  We import torch WITHOUT mujoco -> NO segfault gotcha (there's no mujoco in this file).
"""
import os
import numpy as np

# --- SB3/torch: without mujoco, safe ---
try:
    from stable_baselines3 import PPO
    _SB3_VAR = True
except Exception as e:  # noqa: BLE001
    _SB3_VAR = False
    _SB3_HATA = e

# --- ROS2: MAY NOT be installed on this machine -> exit with an explanatory message ---
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState, Imu
    from std_msgs.msg import Float64MultiArray
    _ROS_VAR = True
except Exception as e:  # noqa: BLE001
    _ROS_VAR = False
    _ROS_HATA = e


# DengeEnv actuator/joint ORDER — we'll build the observation and command in this order.
# (exactly the same as the actuator block in models/mini_humanoid.xml.)
EKLEM_SIRASI = [
    "l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee", "l_ankle_pitch", "l_ankle_roll",
    "r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee", "r_ankle_pitch", "r_ankle_roll",
    "l_shoulder_pitch", "l_shoulder_roll", "l_elbow",
    "r_shoulder_pitch", "r_shoulder_roll", "r_elbow",
    "neck_yaw", "neck_pitch",
]
N_EKLEM = len(EKLEM_SIRASI)          # 20
OBS_BOYUT = 1 + 4 + 6 + N_EKLEM + N_EKLEM   # = 51
CTRL_SCALE = 0.5                     # same as DengeEnv: action[-1,1] -> +-0.5 rad target

VARSAYILAN_MODEL = os.path.join(
    os.path.dirname(__file__), "..", "models", "policy_denge.zip"
)


class PolicyRunner(Node if _ROS_VAR else object):
    """Node that runs the PPO policy at 50 Hz.  Builds the observation from subscriptions, publishes commands."""

    def __init__(self, model_yolu=VARSAYILAN_MODEL, hz=50.0):
        if not _ROS_VAR:
            raise RuntimeError("rclpy not available — this class only runs in a ROS2 environment.")
        super().__init__("policy_runner")

        # ---- 1) LOAD THE BRAIN ----
        if not _SB3_VAR:
            raise RuntimeError(f"stable_baselines3 not available: {_SB3_HATA}")
        self.get_logger().info(f"Loading policy: {model_yolu}")
        self.model = PPO.load(os.path.abspath(model_yolu), device="cpu")

        # ---- 2) SENSOR BUFFERS (latest received messages) ----
        self._q   = np.zeros(N_EKLEM, dtype=np.float64)   # joint angles
        self._qd  = np.zeros(N_EKLEM, dtype=np.float64)   # joint velocities
        self._quat = np.array([1.0, 0.0, 0.0, 0.0])       # body orientation (w,x,y,z)
        self._gyro = np.zeros(3, dtype=np.float64)        # body angular velocity
        self._torso_z = 0.365                             # TODO: real height from odometry/TF
        self._torso_lin = np.zeros(3, dtype=np.float64)   # TODO: real linear velocity from odometry
        self._joint_hazir = False

        # joint_states name -> index mapping (message order may be alphabetical!)
        self._isim2idx = {ad: i for i, ad in enumerate(EKLEM_SIRASI)}

        # ---- 3) SUBSCRIPTIONS ----
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)
        self.create_subscription(Imu, "/imu", self._imu_cb, 10)

        # ---- 4) COMMAND PUBLISHER ----
        # Simple skeleton: a single Float64MultiArray topic (20 target angles, in EKLEM_SIRASI order).
        # In reality this connects to the JointGroupPositionController /commands topic.
        self.cmd_pub = self.create_publisher(
            Float64MultiArray, "/position_controller/commands", 10
        )

        # ---- 5) CONTROL LOOP (50 Hz = DengeEnv control frequency) ----
        self.create_timer(1.0 / hz, self._dongu)
        self.get_logger().info("policy_runner ready. Waiting for /joint_states + /imu...")

    # ---------- SENSOR CALLBACKS ----------
    def _joint_cb(self, msg):
        """/joint_states -> reorder the joints into EKLEM_SIRASI order."""
        for i, ad in enumerate(msg.name):
            j = self._isim2idx.get(ad)
            if j is None:
                continue
            self._q[j] = msg.position[i] if i < len(msg.position) else 0.0
            self._qd[j] = msg.velocity[i] if i < len(msg.velocity) else 0.0
        self._joint_hazir = True

    def _imu_cb(self, msg):
        """/imu -> body orientation (quaternion) + angular velocity."""
        o = msg.orientation
        # ROS quaternion order (x,y,z,w) -> MuJoCo/DengeEnv order (w,x,y,z)
        self._quat = np.array([o.w, o.x, o.y, o.z], dtype=np.float64)
        g = msg.angular_velocity
        self._gyro = np.array([g.x, g.y, g.z], dtype=np.float64)

    # ---------- OBSERVATION BUILDING (SAME ORDER as DengeEnv._get_obs) ----------
    def _gozlem_kur(self):
        # torso velocity (6): [linear(3), angular(3)].  Linear = TODO(odometry), angular = gyro.
        torso_vel = np.concatenate([self._torso_lin, self._gyro])
        obs = np.concatenate([
            np.array([self._torso_z]),   # qpos[2:3]
            self._quat,                  # qpos[3:7]  (w,x,y,z)
            torso_vel,                   # qvel[0:6]
            self._q,                     # qpos[7:27]
            self._qd,                    # qvel[6:26]
        ]).astype(np.float32)
        assert obs.shape[0] == OBS_BOYUT, f"observation {obs.shape[0]} != {OBS_BOYUT}"
        return obs

    # ---------- MAIN LOOP ----------
    def _dongu(self):
        if not self._joint_hazir:
            return  # no sensor data yet
        obs = self._gozlem_kur()
        aksiyon, _ = self.model.predict(obs, deterministic=True)   # 20 x [-1,1]
        aksiyon = np.clip(aksiyon, -1.0, 1.0)
        hedef_aci = aksiyon * CTRL_SCALE                            # same scale as DengeEnv

        msg = Float64MultiArray()
        msg.data = [float(x) for x in hedef_aci]
        self.cmd_pub.publish(msg)


def main(args=None):
    # No ROS: don't crash the program, print an explanatory message.
    if not _ROS_VAR:
        print("[policy_runner] ROS2 (rclpy) not found -> this node only runs in a ROS2 Humble "
              "environment.\n  Error:", _ROS_HATA)
        print("  See gazebo/README.md for setup. (source /opt/ros/humble/setup.bash)")
        return
    if not _SB3_VAR:
        print("[policy_runner] stable_baselines3 not found:", _SB3_HATA)
        return

    rclpy.init(args=args)
    dugum = PolicyRunner()
    try:
        rclpy.spin(dugum)
    except KeyboardInterrupt:
        pass
    finally:
        dugum.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
