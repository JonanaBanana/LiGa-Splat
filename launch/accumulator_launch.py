from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import Node
import os


def generate_launch_description():
    pkg_dir = get_package_share_directory('airlab_lidar_3dgs')

    """Parameters for the composed nodes"""
    in_topic = '/isaacsim/lidar'
    accumulated_topic = '/airlab_lidar_3dgs/accumulated_point_cloud'
    global_topic = '/airlab_lidar_3dgs/global_point_cloud'
    odom_topic = '/isaacsim/odom'
    path_topic = '/airlab_lidar_3dgs/path'

    frame_id = 'World'

    leaf_size = 0.02

    output_location = '/home/airlab/dataset/airlab_3dgs/test1/global_point_cloud.pcd'

    """Generate launch description with multiple components."""
    container = ComposableNodeContainer(
            name='pointcloud_accumulator_container_mt',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='airlab_lidar_3dgs',
                    plugin='airlab_lidar_3dgs::PointCloudAccumulator',
                    name='accumulator_node',
                    parameters=[{
                        'input_topic': in_topic,
                        'output_topic': accumulated_topic,
                        'frame_id': frame_id
                    }],
                    extra_arguments=[{'use_intra_process_comms': True}]
                ),
                ComposableNode(
                    package='airlab_lidar_3dgs',
                    plugin='airlab_lidar_3dgs::GlobalProcessor',
                    name='global_processor_node',
                    parameters=[{
                        'input_topic': accumulated_topic,
                        'output_topic': global_topic,
                        'frame_id': frame_id,
                        'leaf_size': leaf_size,
                        'output_location': output_location
                    }],
                    extra_arguments=[{'use_intra_process_comms': True}]
                ),
                ComposableNode(
                    package='airlab_lidar_3dgs',
                    plugin='airlab_lidar_3dgs::PathPublisher',
                    name='path_publisher_node',
                    parameters=[{
                        'odom_topic': odom_topic,
                        'path_topic': path_topic,
                        'frame_id': frame_id
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
    