#!/usr/bin/env python3
"""
mini_humanoid.launch.py — Launch SKELETON that brings the robot up in Gazebo.

WHAT IT DOES (in order):
  1) Start Gazebo Classic (gazebo_ros).
  2) Feed the URDF to robot_state_publisher (TF + /robot_description).
  3) Spawn the robot into Gazebo (z=0.4 -> feet close to the ground).
  4) Load joint_state_broadcaster + position_controller via controller_manager.
  5) (optional) Start the policy_runner node.

NOTE (honesty): This file is a SYNTAX/STRUCTURE skeleton.  To run it you need a real ROS2
package (mini_humanoid_bringup) + config/controllers.yaml + an install/share layout.
ROS2 MAY NOT be installed on this machine (see the README).  'ros2 launch' only runs this
file inside a ROS2 Humble environment.
"""
# ROS2 launch APIs — if not installed the import blows up; explained in the README.
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, DeclareLaunchArgument, RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Package name — in a real deployment a ROS2 package is created under this name.
    pkg = "mini_humanoid_bringup"
    urdf_yolu = PathJoinSubstitution([FindPackageShare(pkg), "urdf", "mini_humanoid.urdf"])
    # the controller config file is read by the gazebo_ros2_control plugin inside the URDF
    # (config/controllers.yaml).  No need to pass it separately here; kept for reference.

    kullan_sim = DeclareLaunchArgument("use_sim_time", default_value="true")

    # --- 1) Gazebo Classic ---
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("gazebo_ros"), "launch", "gazebo.launch.py"])
        )
    )

    # --- 2) robot_state_publisher: URDF -> /robot_description + TF ---
    # reading the xacro/urdf with 'cat' and putting it into a parameter is a common method; here,
    # instead of reading the file path with Command, as a skeleton we point at robot_description.
    # We read the plain URDF with 'cat' and put it into the /robot_description parameter.
    # (if xacro were used:  Command(["xacro ", urdf_yolu])  )
    robot_description = ParameterValue(Command(["cat ", urdf_yolu]), value_type=str)
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "robot_description": robot_description,
        }],
    )

    # --- 3) Spawn the robot into Gazebo ---
    spawn = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic", "robot_description",
            "-entity", "mini_humanoid",
            "-z", "0.40",           # starting height (feet close to the ground)
        ],
        output="screen",
    )

    # --- 4) Controllers (loaded once spawn finishes) ---
    jsb = Node(
        package="controller_manager", executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )
    pos_ctrl = Node(
        package="controller_manager", executable="spawner",
        arguments=["position_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )
    # Order: spawn -> joint_state_broadcaster -> position_controller
    jsb_sonra = RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb]))
    pos_sonra = RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[pos_ctrl]))

    # --- 5) (optional) policy runner ---
    # policy = Node(package="mini_humanoid_bringup", executable="policy_runner", output="screen")

    return LaunchDescription([
        kullan_sim,
        gazebo,
        rsp,
        spawn,
        jsb_sonra,
        pos_sonra,
        # policy,
    ])
