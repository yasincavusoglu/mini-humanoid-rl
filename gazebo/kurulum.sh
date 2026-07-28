#!/usr/bin/env bash
# ============================================================
# kurulum.sh — Gazebo + ros2_control kurulumu (ROS2 Humble)
# ============================================================
# BU SCRIPTI SEN BURADAYKEN CALISTIR (sudo sorar, ~2 GB indirir).
#   - GUI (gorsel) istiyorsan ONCE bir REBOOT at -> bozuk NVIDIA surucusu duzelir.
#   - Sadece headless fizik dogrulamasi istiyorsan reboot SART DEGIL.
# ============================================================
set -e

echo ">> [1/2] Gazebo Classic + ros2_control paketleri kuruluyor..."
sudo apt-get update
sudo apt-get install -y \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-gazebo-ros2-control \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-xacro

echo ">> [2/2] Dogrulama:"
source /opt/ros/humble/setup.bash
which gazebo && gazebo --version | head -1
ros2 pkg list | grep -E "gazebo_ros2_control|controller_manager" && echo "  ros2_control HAZIR"

echo ""
echo ">> Kurulum tamam. Sonraki adim (birlikte): "
echo "   1) URDF'e gazebo_ros2_control eklentisini bagla (gazebo/README.md'de sablon)"
echo "   2) mini_humanoid_bringup ROS2 paketi + config/controllers.yaml olustur"
echo "   3) gz headless spawn + ros2/policy_runner_numpy.py ile politikayi kostur"
echo "   Detayli plan: SABAH_RAPORU.md (bolum 'Gazebo'yu bitirmek')"
