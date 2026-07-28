#!/usr/bin/env python3
"""
engel_kacis_stub.py — The SIMPLEST obstacle-avoidance reflex (SKELETON + comments).

IDEA:   A depth/lidar sensor reports the NEAREST distance in front of the robot.
        distance < threshold  ->  STOP and TURN (safety reflex).
        Otherwise             ->  let the walking policy run (here just a 'continue' signal).

This is an example of a 'safety layer' running ON TOP OF the RL walking policy:
in a real system this layer can override the output of policy_runner.
The ROS imports are wrapped in try/except — without ROS the node isn't created, so the pure logic can still be tested.
"""
import math

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
    from geometry_msgs.msg import Twist
    _ROS_VAR = True
except Exception as e:  # noqa: BLE001
    _ROS_VAR = False
    _ROS_HATA = e

ESIK_M = 0.30   # obstacle closer than 30 cm -> stop/turn


def karar_ver(min_mesafe, esik=ESIK_M):
    """
    Pure logic (testable without ROS).
    Returns: (forward_speed, turn_speed).
      obstacle near  -> (0.0, +0.5)  = cut forward motion, turn in place
      no obstacle    -> (0.2, 0.0)   = slow forward
    """
    if min_mesafe is None or math.isinf(min_mesafe):
        return 0.2, 0.0            # no data/clear -> slow forward
    if min_mesafe < esik:
        return 0.0, 0.5            # OBSTACLE -> stop + turn
    return 0.2, 0.0               # clear -> forward


if _ROS_VAR:
    class EngelKacis(Node):
        """Listen to /scan (LaserScan) -> publish /cmd_vel (Twist)."""

        def __init__(self):
            super().__init__("engel_kacis")
            self.create_subscription(LaserScan, "/scan", self._scan_cb, 10)
            self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        def _scan_cb(self, msg):
            # Find the smallest valid (non-infinite/non-NaN) distance
            gecerli = [r for r in msg.ranges if r > 0.0 and not math.isinf(r) and not math.isnan(r)]
            min_mesafe = min(gecerli) if gecerli else None
            ileri, don = karar_ver(min_mesafe)
            t = Twist()
            t.linear.x = ileri
            t.angular.z = don
            self.cmd_pub.publish(t)


def main(args=None):
    if not _ROS_VAR:
        print("[engel_kacis] no ROS2 -> pure logic test:")
        for d in [None, 1.2, 0.5, 0.30, 0.15]:
            print(f"  distance={d} -> (fwd,turn)={karar_ver(d)}")
        return
    rclpy.init(args=args)
    dugum = EngelKacis()
    try:
        rclpy.spin(dugum)
    except KeyboardInterrupt:
        pass
    finally:
        dugum.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
