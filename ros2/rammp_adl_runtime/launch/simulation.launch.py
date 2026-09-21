"""Launch the fixture executor with hardware motion disabled unconditionally."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    project_root = LaunchConfiguration("project_root")
    return LaunchDescription([
        DeclareLaunchArgument(
            "project_root", default_value=FindPackageShare("rammp_adl_runtime"),
            description="Installed catalog/schema/config root or explicit source checkout.",
        ),
        DeclareLaunchArgument(
            "context_path", default_value=PathJoinSubstitution([project_root, "examples", "cabinet.context.json"]),
            description="Explicit synthetic scene context for fixture execution.",
        ),
        DeclareLaunchArgument("enable_astra", default_value="false", choices=["true", "false"]),
        DeclareLaunchArgument("time_scale", default_value="0.02"),
        Node(
            package="rammp_adl_runtime", executable="rammp_adl_node", name="rammp_adl",
            output="screen",
            parameters=[{
                "project_root": ParameterValue(project_root, value_type=str),
                "context_path": ParameterValue(LaunchConfiguration("context_path"), value_type=str),
                "enable_astra": ParameterValue(LaunchConfiguration("enable_astra"), value_type=bool),
                "time_scale": ParameterValue(LaunchConfiguration("time_scale"), value_type=float),
                "hardware_motion_enabled": False,
            }],
        ),
    ])
