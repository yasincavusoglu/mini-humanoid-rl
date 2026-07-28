#!/usr/bin/env python3
"""
gazebo_yuru_test.py — Run the DR policy in Gazebo + measure HOW FAR the robot WALKED.
TORCH-FREE (numpy brain). Body state comes via /odom (P3D).
Run for 20 s, print forward distance + min height (the sim-to-real truth).
"""
import os, sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from politika_numpy import NumpyPolitika

EKLEM = ["l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee", "l_ankle_pitch", "l_ankle_roll",
         "r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee", "r_ankle_pitch", "r_ankle_roll",
         "l_shoulder_pitch", "l_shoulder_roll", "l_elbow", "r_shoulder_pitch", "r_shoulder_roll",
         "r_elbow", "neck_yaw", "neck_pitch"]


class Test(Node):
    def __init__(self, sure=20.0):
        super().__init__("gazebo_yuru_test")
        ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        npz = os.environ.get("POLITIKA_NPZ", f"{ROOT}/models/politika_yuruyus_dr_numpy.npz")
        self.pol = NumpyPolitika(npz)
        self.get_logger().info(f"DR policy (numpy): {npz}")
        self.jpos, self.jvel = {}, {}
        self.quat = np.array([1., 0, 0, 0]); self.wvel = np.zeros(3)
        self.z = 0.36; self.lvel = np.zeros(3); self._ok = False
        self.x0 = None; self.x = 0.0; self.zmin = 1.0; self.t0 = None; self.sure = sure
        self.create_subscription(JointState, "/joint_states", self.on_j, 10)
        self.create_subscription(Imu, "/imu", self.on_i, 10)
        self.create_subscription(Odometry, "/odom", self.on_o, 10)
        self.pub = self.create_publisher(Float64MultiArray, "/position_controller/commands", 10)
        self.create_timer(0.02, self.step)

    def on_j(self, m):
        for n, p in zip(m.name, m.position): self.jpos[n] = p
        for n, v in zip(m.name, m.velocity): self.jvel[n] = v
        self._ok = all(n in self.jpos for n in EKLEM)

    def on_i(self, m):
        q = m.orientation; self.quat = np.array([q.w, q.x, q.y, q.z])
        w = m.angular_velocity; self.wvel = np.array([w.x, w.y, w.z])

    def on_o(self, m):
        p = m.pose.pose.position; self.z = p.z; self.x = p.x
        v = m.twist.twist.linear; self.lvel = np.array([v.x, v.y, v.z])
        if self.x0 is None: self.x0 = self.x

    def step(self):
        if not self._ok:
            return
        if self.t0 is None: self.t0 = time.time()
        jp = np.array([self.jpos[n] for n in EKLEM])
        jv = np.array([self.jvel.get(n, 0.0) for n in EKLEM])
        obs = np.concatenate([[self.z], self.quat, self.lvel, self.wvel, jp, jv]).astype(np.float64)
        a = self.pol.act(obs)
        msg = Float64MultiArray(); msg.data = [float(x) * 0.5 for x in a]
        self.pub.publish(msg)
        self.zmin = min(self.zmin, self.z)
        if time.time() - self.t0 > self.sure:
            dx = (self.x - self.x0) if self.x0 is not None else 0.0
            durum = "FELL" if self.zmin < 0.20 else "UPRIGHT"
            print(f"RESULT: {self.sure:.0f}s -> forward {dx:+.2f} m | min height {self.zmin:.2f} m | {durum}", flush=True)
            rclpy.shutdown()


def main():
    rclpy.init()
    rclpy.spin(Test(20.0))


if __name__ == "__main__":
    main()
