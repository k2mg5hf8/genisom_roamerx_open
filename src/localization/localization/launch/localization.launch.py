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
    gnss_recovery_disabled_file = os.path.join(
        localization_dir, 'config', 'gnss_recovery_disabled.yaml')

    # Per-robot sensor topic names. Defaults match config.yaml's points_topic/imu_topic,
    # so on the "native" robot nothing changes. On a robot with different native topic
    # names (e.g. Livox driver publishing /livox/lidar, /livox/imu), pass these as launch
    # args instead of running `ros2 run topic_tools relay ...` - remapping happens at
    # subscription time, no extra process, no duplicated sensor traffic.
    lidar_topic_arg = DeclareLaunchArgument('lidar_topic', default_value='/front_lidar')
    imu_topic_arg = DeclareLaunchArgument('imu_topic', default_value='/front_lidar/imu')
    gnss_recovery_config_arg = DeclareLaunchArgument(
        'gnss_recovery_config',
        default_value=gnss_recovery_disabled_file,
        description=(
            'Optional site-specific GNSS recovery parameter file. The default '
            'keeps GNSS recovery completely disabled.'
        ),
    )

    return LaunchDescription([
        lidar_topic_arg,
        imu_topic_arg,
        gnss_recovery_config_arg,
        Node(
            package='localization',
            executable='pointcloud_self_filter_node',
            name='pointcloud_self_filter',
            output='screen',
            parameters=[
                config_file,
                {'input_topic': LaunchConfiguration('lidar_topic')},
            ],
        ),
        Node(
            package='localization',
            executable='motor_odom_deduplicator_node',
            name='motor_odom_deduplicator',
            output='screen',
            parameters=[config_file],
        ),
        # 启动定位节点
        Node(
            package='localization',
            executable='localization_node',
            name='localization',
            output='screen',
            parameters=[config_file, LaunchConfiguration('gnss_recovery_config')],
            remappings=[
                ('/front_lidar', LaunchConfiguration('lidar_topic')),
                ('/front_lidar/imu', LaunchConfiguration('imu_topic')),
                # Isolate the MO dynamic TF authority from the closed
                # /robot_tf selector that continues publishing on /tf.
                ('/tf', '/mo_tf'),
                ('/tf_static', '/mo_tf_static'),
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
            arguments=[
                '--x', '0.0', '--y', '0.0', '--z', '0.0',
                '--qx', '0.0', '--qy', '0.0', '--qz', '0.0', '--qw', '1.0',
                '--frame-id', 'base_link', '--child-frame-id', 'livox_frame',
            ],
            remappings=[('/tf_static', '/mo_tf_static')],
        )
    ])
