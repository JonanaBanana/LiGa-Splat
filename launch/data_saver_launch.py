from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import Node
import os
import yaml


def generate_launch_description():
    pkg_dir = get_package_share_directory('liga_splat')

    with open(os.path.join(pkg_dir, 'config', 'launch_config_test.cfg'), 'r') as f:
        cfg = yaml.safe_load(f)

    output_dir         = cfg['output_dir']
    lidar_topic        = cfg['lidar_topic']
    odom_topic         = cfg['odom_topic']
    image_topic        = cfg['image_topic']
    image_save_interval = cfg['image_save_interval']
    image_prefix       = cfg['image_prefix']
    odom_save_interval = cfg['odom_save_interval']
    frame_id           = cfg['frame_id']
    leaf_size             = cfg['leaf_size']
    max_path_length       = cfg['max_path_length']
    accumulator_max_points    = cfg['accumulator_max_points']
    accumulator_publish_interval = cfg['accumulator_publish_interval']
    global_downsample_interval   = cfg['global_downsample_interval']
    global_max_points            = cfg['global_max_points']

    accumulated_topic = '/liga_splat/accumulated_point_cloud'
    global_topic      = '/liga_splat/global_point_cloud'
    path_topic        = '/liga_splat/path'

    #Subdirectories
    #Do not change these subdirectories as they are used by the nodes to save data in an organized manner.
    pcd_output_dir = os.path.join(output_dir, 'pcd')
    image_output_dir = os.path.join(output_dir, 'distorted/images')
    timestamp_dir = os.path.join(output_dir, 'timestamps')
    
    #Output File Locations
    pcd_output_location = os.path.join(pcd_output_dir, 'input.pcd')
    image_timestamp_output_location = os.path.join(timestamp_dir, 'image.csv')
    odom_timestamp_output_location = os.path.join(timestamp_dir, 'odom.csv')

    #Create necessary directories if they don't exist.
    os.makedirs(pcd_output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(timestamp_dir, exist_ok=True)

    #Generate launch description with multiple components.
    container = ComposableNodeContainer(
            name='pointcloud_accumulator_container_mt',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='liga_splat',
                    plugin='liga_splat::PointCloudAccumulator',
                    name='accumulator_node',
                    parameters=[{
                        'input_topic': lidar_topic,
                        'output_topic': accumulated_topic,
                        'frame_id': frame_id,
                        'max_points': accumulator_max_points,
                        'publish_interval': accumulator_publish_interval,
                        'leaf_size': leaf_size
                    }],
                    extra_arguments=[{'use_intra_process_comms': True}]
                ),
                ComposableNode(
                    package='liga_splat',
                    plugin='liga_splat::GlobalProcessor',
                    name='global_processor_node',
                    parameters=[{
                        'input_topic': accumulated_topic,
                        'output_topic': global_topic,
                        'frame_id': frame_id,
                        'leaf_size': leaf_size,
                        'output_location': pcd_output_location,
                        'downsample_interval': global_downsample_interval,
                        'max_global_points': global_max_points
                    }],
                    extra_arguments=[{'use_intra_process_comms': True}]
                ),
                ComposableNode(
                    package='liga_splat',
                    plugin='liga_splat::PathPublisher',
                    name='path_publisher_node',
                    parameters=[{
                        'odom_topic': odom_topic,
                        'path_topic': path_topic,
                        'frame_id': frame_id,
                        'max_path_length': max_path_length
                    }],
                    extra_arguments=[{'use_intra_process_comms': True}]
                ),
                ComposableNode(
                    package='liga_splat',
                    plugin='liga_splat::ImageSaver',
                    name='image_saver_node',
                    parameters=[{
                        'image_topic': image_topic,
                        'save_interval': image_save_interval,
                        'image_dir': image_output_dir,
                        'timestamp_file': image_timestamp_output_location,
                        'image_prefix': image_prefix
                    }],
                    extra_arguments=[{'use_intra_process_comms': True}]
                ),
                ComposableNode(
                    package='liga_splat',
                    plugin='liga_splat::OdomSaver',
                    name='odom_saver_node',
                    parameters=[{
                        'odom_topic': odom_topic,
                        'output_file': odom_timestamp_output_location,
                        'save_interval': odom_save_interval
                    }],
                    extra_arguments=[{'use_intra_process_comms': True}]
                ),
            ],
            output='screen',
    )
    return LaunchDescription([container, 
                              Node(
                                package='rviz2',
                                executable='rviz2',
                                name='rviz2',
                                arguments=['-d', os.path.join(pkg_dir, 'rviz', 'pointcloud.rviz')]
                                )
                            ])
    