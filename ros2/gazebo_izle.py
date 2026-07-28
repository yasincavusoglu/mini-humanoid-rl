#!/usr/bin/env python3
"""
gazebo_izle.py — Record the z (height) + x (forward) trace of mini_humanoid in Gazebo over a DURATION.
Purpose: while the policy runs, is the robot UPRIGHT (z~0.3-0.4), WALKING (x increases), or has it FALLEN (z<0.25)?
Usage: python3 ros2/gazebo_izle.py --sure 15
"""
import sys, time, argparse
import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates


class Izle(Node):
    def __init__(self, sure):
        super().__init__("gazebo_izle")
        self.sure = sure
        self.t0 = None
        self.zs, self.xs, self.ts = [], [], []
        self.create_subscription(ModelStates, "/model_states", self.cb, 10)

    def cb(self, m):
        try:
            i = m.name.index("mini_humanoid")
        except ValueError:
            return
        t = time.time()
        if self.t0 is None:
            self.t0 = t
        self.zs.append(m.pose[i].position.z)
        self.xs.append(m.pose[i].position.x)
        self.ts.append(t - self.t0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sure", type=float, default=15.0)
    args = ap.parse_args()
    rclpy.init()
    n = Izle(args.sure)
    t_start = time.time()
    while time.time() - t_start < args.sure:
        rclpy.spin_once(n, timeout_sec=0.2)
    if not n.zs:
        print("RESULT: mini_humanoid NEVER arrived from /model_states (spawn/plugin issue?)")
    else:
        z0, zmin, zson = n.zs[0], min(n.zs), n.zs[-1]
        x0, xson = n.xs[0], n.xs[-1]
        dustu = zmin < 0.25
        print(f"RESULT ({len(n.zs)} samples, {n.ts[-1]:.1f}s):")
        print(f"  height z: start {z0:.3f} -> min {zmin:.3f} -> last {zson:.3f} m")
        print(f"  forward  x : {x0:.3f} -> {xson:.3f}  (traveled {xson-x0:+.3f} m)")
        print(f"  STATUS: {'FELL (z<0.25)' if dustu else 'stayed UPRIGHT'}"
              f"{' + ADVANCED' if (xson-x0) > 0.05 else ''}")
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
