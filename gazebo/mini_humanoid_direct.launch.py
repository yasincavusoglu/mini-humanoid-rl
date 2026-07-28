#!/usr/bin/env python3
"""
mini_humanoid_direct.launch.py — Direct launch WITHOUT COLCON (absolute paths).
mini_humanoid.launch.py requires FindPackageShare(mini_humanoid_bringup) (needs package+build).
This version reads the files from the repo via ABSOLUTE paths -> runs directly with `ros2 launch`.

Usage:
  source /opt/ros/humble/setup.bash
  ros2 launch gazebo/mini_humanoid_direct.launch.py gui:=false politika:=true

Flow: Gazebo(+ros_state -> /model_states) -> RSP(URDF) -> spawn(z=0.40) ->
      joint_state_broadcaster -> position_controller -> (delayed) yuru10 policy runner.
"""
import os
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, DeclareLaunchArgument, RegisterEventHandler,
    ExecuteProcess, TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

ROOT = "/home/yasin/Workspace/humanoid_rl"
URDF = os.path.join(ROOT, "gazebo", "mini_humanoid.urdf")
NPZ = os.path.join(ROOT, "models", "politika_yuru10_numpy.npz")


def generate_launch_description():
    gui = DeclareLaunchArgument("gui", default_value="false")
    politika = DeclareLaunchArgument("politika", default_value="true")

    # --- 1) Gazebo Classic (headless selectable) + ros_state plugin (-> /model_states) ---
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("gazebo_ros"), "launch", "gazebo.launch.py"])
        ),
        launch_arguments={
            "gui": LaunchConfiguration("gui"),
            "verbose": "true",
            "extra_gazebo_args": "-s libgazebo_ros_state.so",  # so /model_states gets published
        }.items(),
    )

    # --- 2) robot_state_publisher: URDF (cat) -> /robot_description + TF ---
    robot_description = ParameterValue(Command(["cat ", URDF]), value_type=str)
    rsp = Node(
        package="robot_state_publisher", executable="robot_state_publisher", output="screen",
        parameters=[{"use_sim_time": True, "robot_description": robot_description}],
    )

    # --- 3) Spawn the robot (z=0.40) ---
    spawn = Node(
        package="gazebo_ros", executable="spawn_entity.py",
        arguments=["-topic", "robot_description", "-entity", "mini_humanoid", "-z", "0.40"],
        output="screen",
    )

    # --- 4) Controllers (in order: spawn -> jsb -> position) ---
    jsb = Node(package="controller_manager", executable="spawner",
               arguments=["joint_state_broadcaster", "-c", "/controller_manager"], output="screen")
    pos_ctrl = Node(package="controller_manager", executable="spawner",
                    arguments=["position_controller", "-c", "/controller_manager"], output="screen")
    jsb_sonra = RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb]))
    pos_sonra = RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[pos_ctrl]))

    # --- 5) yuru10 policy runner (SYSTEM python3 + numpy; NO torch). 3s after pos_ctrl. ---
    policy = TimerAction(period=3.0, actions=[
        ExecuteProcess(
            cmd=["python3", os.path.join(ROOT, "ros2", "gazebo_policy_runner.py")],
            additional_env={"POLITIKA_NPZ": NPZ, "STATE_TOPIC": "/model_states"},
            output="screen",
            condition=IfCondition(LaunchConfiguration("politika")),
        )
    ])
    policy_sonra = RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[policy]))

    return LaunchDescription([gui, politika, gazebo, rsp, spawn, jsb_sonra, pos_sonra, policy_sonra])
