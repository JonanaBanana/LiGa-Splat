from launch import LaunchDescription
from launch.actions import ExecuteProcess
import os
from datetime import datetime

def generate_launch_description():

    bag_name = 'isaacsim_bag_' + datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('/home/airlab/dataset/isaacsim', bag_name)

    topics = [
        '/isaacsim/camera_info',
        '/isaacsim/rgb',
        '/isaacsim/lidar',
        '/isaacsim/imu',
        '/isaacsim/odom',
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