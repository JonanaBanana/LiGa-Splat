#!/bin/bash 
if [ $# -eq 0 ]
  then
    echo "No data path supplied, exiting!"
    echo "Example usage: ./full_pipeline.sh /path/to/data_folder"
    exit 1
fi
file="$1"
source /home/airlab/ros2_ws/install/setup.bash
ros2 run airlab_lidar_3dgs pose_estimator "$file"
wait
ros2 run airlab_lidar_3dgs registration "$file" --diag
wait
ros2 run airlab_lidar_3dgs reconstruction "$file"
wait
ros2 run airlab_lidar_3dgs export_colmap "$file"
wait
ros2 run airlab_lidar_3dgs depth_renderer "$file" --diag --dense
wait
ros2 run airlab_lidar_3dgs prepare_depth_for_3dgs "$file"