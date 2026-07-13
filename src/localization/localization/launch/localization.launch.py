import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # 获取配置文件路径
    localization_dir = get_package_share_directory('localization')
    config_file = os.path.join(localization_dir, 'config', 'config.yaml')

    # Per-robot sensor topic names. Defaults match config.yaml's points_topic/imu_topic,
    # so on the "native" robot nothing changes. On a robot with different native topic
    # names (e.g. Livox driver publishing /livox/lidar, /livox/imu), pass these as launch
    # args instead of running `ros2 run topic_tools relay ...` - remapping happens at
    # subscription time, no extra process, no duplicated sensor traffic.
    lidar_topic_arg = DeclareLaunchArgument('lidar_topic', default_value='/front_lidar')
    imu_topic_arg = DeclareLaunchArgument('imu_topic', default_value='/front_lidar/imu')

    return LaunchDescription([
        lidar_topic_arg,
        imu_topic_arg,
        # 启动定位节点
        Node(
            package='localization',
            executable='localization_node',
            name='localization',
            output='screen',
            parameters=[config_file],
            remappings=[
                ('/front_lidar', LaunchConfiguration('lidar_topic')),
                ('/front_lidar/imu', LaunchConfiguration('imu_topic')),
            ]
        ),

        # Node(
        #     package='localization',
        #     executable='localization_node',
        #     name='localization_map_server',
        #     output='screen',
        #     parameters=[config_file]
        # )
        
        # 静态TF发布器
        Node(
            name='lidar_tf',
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0.0', '0.0', '0.0', '0', '0', '0', '1', 'base_link', 'livox_frame']
        )
    ]) 