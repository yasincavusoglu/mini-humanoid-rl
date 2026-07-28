#!/usr/bin/env python3
"""
gazebo_hold_yuru.py — Gazebo sim-to-real DIAGNOSTIC:
  hold NEUTRAL for the first 3 s (can it stay upright?) -> then WALK with the DR policy (is it advancing?).
z (height) and x (forward) are printed periodically; RESULT at the end. Torch-free (numpy).
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


class T(Node):
    def __init__(self):
        super().__init__("gazebo_hold_yuru")
        ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.pol = NumpyPolitika(f"{ROOT}/models/politika_yuruyus_dr_numpy.npz")
        self.jpos, self.jvel = {}, {}
        self.quat = np.array([1., 0, 0, 0]); self.wvel = np.zeros(3)
        self.z = 0.36; self.lvel = np.zeros(3)
        self._ok = False; self._printed = False
        self.x0 = None; self.x = 0.; self.zmin = 1.; self.t0 = None; self.n = 0
        self.create_subscription(JointState, "/joint_states", self.on_j, 10)
        self.create_subscription(Imu, "/imu", self.on_i, 10)
        self.create_subscription(Odometry, "/odom", self.on_o, 10)
        self.pub = self.create_publisher(Float64MultiArray, "/position_controller/commands", 10)
        self.create_timer(0.02, self.step)

    def on_j(self, m):
        for n, p in zip(m.name, m.position): self.jpos[n] = p
        for n, v in zip(m.name, m.velocity): self.jvel[n] = v
        if not self._printed:
            eksik = [n for n in EKLEM if n not in self.jpos]
            self.get_logger().info(f"joint_states: {len(m.name)} joints received; missing from EKLEM={eksik}")
            self._printed = True
        self._ok = all(n in self.jpos for n in EKLEM)

    def on_i(self, m):
        q = m.orientation; self.quat = np.array([q.w, q.x, q.y, q.z])
        self.wvel = np.array([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])

    def on_o(self, m):
        p = m.pose.pose.position; self.z = p.z; self.x = p.x
        v = m.twist.twist.linear; self.lvel = np.array([v.x, v.y, v.z])
        if self.x0 is None: self.x0 = self.x

    def step(self):
        if not self._ok:
            return
        if self.t0 is None: self.t0 = time.time()
        self.n += 1
        t = time.time() - self.t0
        jp = np.array([self.jpos[n] for n in EKLEM])
        jv = np.array([self.jvel.get(n, 0.) for n in EKLEM])
        if t < 3.0:                                   # hold NEUTRAL
            a = np.zeros(20)
        else:                                          # walk with DR
            obs = np.concatenate([[self.z], self.quat, self.lvel, self.wvel, jp, jv]).astype(np.float64)
            a = self.pol.act(obs)
        self.pub.publish(Float64MultiArray(data=[float(x) * 0.5 for x in a]))
        self.zmin = min(self.zmin, self.z)
        if self.n % 50 == 0:
            dx = (self.x - self.x0) if self.x0 is not None else 0.
            self.get_logger().info(f"t={t:4.1f}s  z={self.z:.3f}  fwd={dx:+.2f}m  phase={'NEUTRAL' if t < 3 else 'WALK'}")
        if t > 18:
            dx = (self.x - self.x0) if self.x0 is not None else 0.
            print(f"RESULT: forward {dx:+.2f} m | min z {self.zmin:.2f} | last z {self.z:.2f} | {'FELL' if self.zmin < 0.20 else 'UPRIGHT'}", flush=True)
            rclpy.shutdown()


def main():
    rclpy.init()
    rclpy.spin(T())


if __name__ == "__main__":
    main()
