from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
import os
import yaml
from datetime import datetime

def generate_launch_description():
    pkg_dir = get_package_share_directory('liga_splat')

    with open(os.path.join(pkg_dir, 'config', 'launch_config.cfg'), 'r') as f:
        cfg = yaml.safe_load(f)

    bag_name = 'isaacsim_bag_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(cfg['bag_output_dir'], bag_name)

    topics = [
        '/isaacsim/camera_info',
        cfg['image_topic'],
        cfg['lidar_topic'],
        cfg['odom_topic'],
        '/tf',
    ]

    return LaunchDescription([
        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'record',
                '--output', output_dir,
                '--max-bag-size', '1000000000', # 1 GB splits
            ] + topics,
            output='screen',
            shell=False,
        )
    ])