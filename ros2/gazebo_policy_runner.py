#!/usr/bin/env python3
"""
gazebo_policy_runner.py — ROS2 node that runs the trained walking policy in GAZEBO.
TORCH-FREE (numpy) -> runs in the ROS2 system python.

Input (Gazebo -> subscribe):
  /joint_states (JointState)  -> 20 joint angles + velocities
  /imu          (Imu)         -> body orientation (quaternion) + angular velocity
  /odom         (Odometry)    -> body height(z) + linear velocity  (the IMU doesn't provide this -> P3D)
Output (-> publish):
  /position_controller/commands (Float64MultiArray) -> 20 target joint angles (= action*0.5)

The observation is built in the EXACT same order as Yuru10Env._get_obs (PHASE-CLOCKED, 53-D):
  [z(1), quat wxyz(4), lin_vel(3), ang_vel(3), 20 joint angles, 20 joint velocities, sin(faz), cos(faz)] = 53
  Phase clock: at each 50Hz step phase += freq*DT (freq=0.85, DT=0.02), same rhythm as the sim.
"""
import os, sys
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float64MultiArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from politika_numpy import NumpyPolitika

EKLEM = ["l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee", "l_ankle_pitch", "l_ankle_roll",
         "r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee", "r_ankle_pitch", "r_ankle_roll",
         "l_shoulder_pitch", "l_shoulder_roll", "l_elbow", "r_shoulder_pitch", "r_shoulder_roll",
         "r_elbow", "neck_yaw", "neck_pitch"]
CTRL_SCALE = 0.5


class GazeboPolicyRunner(Node):
    def __init__(self):
        super().__init__("gazebo_policy_runner")
        ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        npz = os.environ.get("POLITIKA_NPZ", f"{ROOT}/models/politika_yuru10_numpy.npz")
        self.pol = NumpyPolitika(npz)
        self.get_logger().info(f"Policy loaded (torch-free): {npz}")

        self.jpos, self.jvel = {}, {}
        self.quat = np.array([1.0, 0, 0, 0]); self.wvel = np.zeros(3)
        self.z = 0.36; self.lvel = np.zeros(3)
        self._j_ok = False
        # PHASE CLOCK (yuru10 = 53-D obs): advances at the same rhythm as the sim
        self._phase = 0.0
        self.freq = 0.85
        self.DT = 0.02

        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)
        self.create_subscription(Imu, "/imu", self.on_imu, 10)
        self.create_subscription(ModelStates, os.environ.get("STATE_TOPIC", "/model_states"), self.on_model_states, 10)
        self.pub = self.create_publisher(Float64MultiArray, "/position_controller/commands", 10)
        self.timer = self.create_timer(0.02, self.step)   # 50 Hz control
        self.get_logger().info("Gazebo policy runner ready (50 Hz).")

    def on_joints(self, m):
        for n, p in zip(m.name, m.position):
            self.jpos[n] = p
        for n, v in zip(m.name, m.velocity):
            self.jvel[n] = v
        self._j_ok = all(n in self.jpos for n in EKLEM)

    def on_imu(self, m):
        q = m.orientation; self.quat = np.array([q.w, q.x, q.y, q.z])
        w = m.angular_velocity; self.wvel = np.array([w.x, w.y, w.z])

    def on_model_states(self, m):
        try:
            i = m.name.index("mini_humanoid")
        except ValueError:
            return
        self.z = m.pose[i].position.z
        v = m.twist[i].linear; self.lvel = np.array([v.x, v.y, v.z])

    def step(self):
        if not self._j_ok:
            return
        jp = np.array([self.jpos[n] for n in EKLEM])
        jv = np.array([self.jvel.get(n, 0.0) for n in EKLEM])
        # advance the phase clock (same as sim: phase += freq*DT), then append sin/cos -> 53-D
        self._phase = (self._phase + self.freq * self.DT) % 1.0
        faz = [np.sin(2 * np.pi * self._phase), np.cos(2 * np.pi * self._phase)]
        obs = np.concatenate([[self.z], self.quat, self.lvel, self.wvel, jp, jv, faz]).astype(np.float64)
        a = self.pol.act(obs)                       # [-1,1]
        msg = Float64MultiArray()
        msg.data = [float(x) * CTRL_SCALE for x in a]   # target joint angle (rad)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = GazeboPolicyRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
