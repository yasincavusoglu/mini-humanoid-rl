"""
gazebo_test.launch.py — Spawn the robot in Gazebo (HEADLESS) + load the ros2_control controllers.
GPU broken -> only gzserver (physics), no GUI. IMU+physics don't need GL.
Flow: gzserver -> robot_state_publisher(URDF) -> spawn -> joint_state_broadcaster -> position_controller
"""
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node

ROOT = "/home/yasin/Workspace/humanoid_rl"
URDF = f"{ROOT}/gazebo/mini_humanoid.urdf"


def generate_launch_description():
    import re
    with open(URDF) as f:
        robot_desc = f.read()
    # gazebo_ros2_control (Humble) blows up the YAML-param parser when passing
    # robot_description as --param. Fix: strip COMMENTS (the ': ' etc. inside them break YAML) +
    # strip the xml declaration + collapse all whitespace to a single space.
    robot_desc = re.sub(r"<!--.*?-->", "", robot_desc, flags=re.DOTALL)
    robot_desc = re.sub(r"<\?xml[^>]*\?>", "", robot_desc)
    robot_desc = re.sub(r"\s+", " ", robot_desc).strip()

    gazebo = ExecuteProcess(
        cmd=["gzserver", "--verbose",
             "-s", "libgazebo_ros_init.so",
             "-s", "libgazebo_ros_factory.so",
             "-s", "libgazebo_ros_state.so"],
        output="screen",
    )
    rsp = Node(
        package="robot_state_publisher", executable="robot_state_publisher", output="screen",
        parameters=[{"robot_description": robot_desc, "use_sim_time": True}],
    )
    spawn = Node(
        package="gazebo_ros", executable="spawn_entity.py", output="screen",
        arguments=["-topic", "robot_description", "-entity", "mini_humanoid", "-z", "0.40"],
    )
    jsb = ExecuteProcess(
        cmd=["ros2", "run", "controller_manager", "spawner", "joint_state_broadcaster"],
        output="screen",
    )
    posc = ExecuteProcess(
        cmd=["ros2", "run", "controller_manager", "spawner", "position_controller"],
        output="screen",
    )

    return LaunchDescription([
        gazebo, rsp, spawn,
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[posc])),
    ])
