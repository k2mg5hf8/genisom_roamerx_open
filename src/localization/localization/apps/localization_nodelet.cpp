// hdl localizaton 
#include <mutex>
#include <memory>
#include <limits>
#include <array>
#include <iostream>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <deque>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <future>
#include <iterator>
#include <cstring>

#include <rclcpp/rclcpp.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <std_srvs/srv/empty.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>

#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>

#include <pclomp/ndt_omp.h>
#include <fast_gicp/ndt/ndt_cuda.hpp>

#include <localization/pose_estimator.hpp>
#include <localization/static_imu_init.hpp>
#include <localization/mode_state.h>
#include <localization/global_localization.hpp>

#include <localization/msg/scan_matching_status.hpp>
#include <robots_dog_msgs/srv/load_map.hpp>
#include <robots_dog_msgs/srv/localization_state.hpp>
#include <robots_dog_msgs/msg/localization.hpp>

using namespace std;

namespace localization {

class HdlLocalizationNode : public rclcpp::Node {
public:
  using PointT = pcl::PointXYZI;

  HdlLocalizationNode(const rclcpp::NodeOptions& options) : Node("localization", options) {
    tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    robot_odom_frame_id              = declare_parameter<std::string>("robot_odom_frame_id", "map");
    odom_child_frame_id              = declare_parameter<std::string>("odom_child_frame_id", "livox_frame");
    send_tf_transforms               = declare_parameter<bool>("send_tf_transforms", false);
    send_odom_base_transform_        = declare_parameter<bool>("send_odom_base_transform", false);
    publish_bootstrap_map_to_odom_   = declare_parameter<bool>("publish_bootstrap_map_to_odom", true);
    cool_time_duration               = declare_parameter<double>("cool_time_duration", 0.5);
    reg_method                       = declare_parameter<std::string>("reg_method", "NDT_OMP");
    ndt_neighbor_search_method       = declare_parameter<std::string>("ndt_neighbor_search_method", "DIRECT7");
    ndt_neighbor_search_radius       = declare_parameter<double>("ndt_neighbor_search_radius", 2.0);
    ndt_resolution                   = declare_parameter<double>("ndt_resolution", 1.0);
    enable_robot_odometry_prediction = declare_parameter<bool>("enable_robot_odometry_prediction", false);
    enable_internal_odom_ukf_ =
      declare_parameter<bool>("enable_internal_odom_ukf", false);

    use_imu     = declare_parameter<bool>("use_imu", true);
    invert_acc  = declare_parameter<bool>("invert_acc", false);
    invert_gyro = declare_parameter<bool>("invert_gyro", false);
    // imu static init params
    imu_init_time_         = static_cast<float>(declare_parameter<double>("imu_init_time", 3.0));
    imu_init_queue_size_   = declare_parameter<int>("imu_init_queue_size", 600);
    imu_init_max_gyro_var_ = static_cast<float>(declare_parameter<double>("imu_init_max_gyro_var", 0.05));
    imu_init_max_acce_var_ = static_cast<float>(declare_parameter<double>("imu_init_max_acce_var", 0.2));

    std::string imu_topic               = declare_parameter<std::string>("imu_topic", "/livox/imu");
    std::string points_topic            = declare_parameter<std::string>("points_topic", "/livox/lidar");
    std::string motor_odom_topic        = declare_parameter<std::string>("motor_odom_topic", "/odom/mc_odom");
    std::string odom_topic              = declare_parameter<std::string>("odom_topic", "/odom/localization_odom");
    std::string local_odom_topic        = declare_parameter<std::string>("local_odom_topic", "/odom/fused_odom");
    std::string aligned_points_topic    = declare_parameter<std::string>("aligned_points_topic", "/aligned_points");
    std::string status_topic            = declare_parameter<std::string>("status_topic", "/status");
    std::string localization_info_topic = declare_parameter<std::string>("localization_info_topic", "/localization_info");
    std::string localization_health_topic =
      declare_parameter<std::string>("localization_health_topic", "/localization_info_health");
    publish_vendor_localization_info_ =
      declare_parameter<bool>("publish_vendor_localization_info", false);
    std::string global_map_points_topic = declare_parameter<std::string>("global_map_points_topic", "/global_map_points");

    // Load numeric parameters
    imu_data_filter_num_      = declare_parameter<int>("imu_data_filter_num", 5);
    imu_buffer_max_size_      = static_cast<size_t>(std::max<int64_t>(
      256, declare_parameter<int>("imu_buffer_max_size", 4096)));
    globalmap_voxel_size_     = static_cast<float>(declare_parameter<double>("globalmap_voxel_size", 0.3));
    points_voxel_filter_size_ = static_cast<float>(declare_parameter<double>("points_voxel_filter_size", 0.30));
    
    min_valid_count_           = declare_parameter<int>("min_valid_count", 5);
    buffer_size_               = declare_parameter<int>("buffer_size", 10);
    localization_odom_frame_id = declare_parameter<std::string>("localization_odom_frame_id", "base_link");

    // IMU rotation matrix parameters 
    std::vector<double> init_imu_R;
    init_imu_R.reserve(9);
    declare_parameter<std::vector<double>>("init_R", std::vector<double>{});
    get_parameter("init_R", init_imu_R);
    
    // Gravity transform matrix parameters 
    std::vector<double> init_T;
    init_T.reserve(16);
    declare_parameter<std::vector<double>>("init_T", std::vector<double>{});
    get_parameter("init_T", init_T);
    
    // Initial pose initialization parameters 
    declare_parameter<int>("init_match_count_threshold", 5);
    get_parameter("init_match_count_threshold", init_match_count_threshold_);
    bad__match_count_threshold_ = declare_parameter<int>("bad_match_count_threshold", 6);
    tracking_degraded_count_threshold_ = std::max(
      1, static_cast<int>(
        declare_parameter<int>("tracking_degraded_count_threshold", 2)));
    declare_parameter<float>("init_match_score_threshold", 0.15);
    get_parameter("init_match_score_threshold", init_match_score_threshold_);
    tracking_max_xy_jump_ = static_cast<float>(declare_parameter<double>("tracking_max_xy_jump", 0.3));
    tracking_max_yaw_jump_deg_ = static_cast<float>(declare_parameter<double>("tracking_max_yaw_jump_deg", 15.0));
    reliable_threshold_ = static_cast<float>(declare_parameter<double>("tracking_max_fitness_score", 0.25));
    planar_ndt_enabled_ = declare_parameter<bool>("planar_ndt_enabled", true);
    localization_scan_min_interval_sec_ =
      declare_parameter<double>("localization_scan_min_interval_sec", 0.18);
    motor_twist_stale_timeout_sec_ =
      declare_parameter<double>("motor_twist_stale_timeout_sec", 0.35);
    motor_twist_filter_alpha_ = static_cast<float>(
      declare_parameter<double>("motor_twist_filter_alpha", 0.25));
    local_odom_publish_rate_hz_ = std::max(
      1.0, declare_parameter<double>("local_odom_publish_rate_hz", 50.0));
    tracking_min_position_stddev_ = declare_parameter<double>(
      "tracking_min_position_stddev", 0.05);
    tracking_min_orientation_stddev_deg_ = declare_parameter<double>(
      "tracking_min_orientation_stddev_deg", 2.0);
    initializing_position_stddev_ = declare_parameter<double>(
      "initializing_position_stddev", 0.50);
    initializing_orientation_stddev_deg_ = declare_parameter<double>(
      "initializing_orientation_stddev_deg", 30.0);
    degraded_position_stddev_ = declare_parameter<double>(
      "degraded_position_stddev", 0.75);
    degraded_orientation_stddev_deg_ = declare_parameter<double>(
      "degraded_orientation_stddev_deg", 45.0);
    fused_prediction_translation_margin_ = static_cast<float>(
      declare_parameter<double>("fused_prediction_translation_margin", 0.05));
    fused_prediction_max_speed_ = static_cast<float>(
      declare_parameter<double>("fused_prediction_max_speed", 0.75));
    fused_prediction_max_yaw_step_deg_ = static_cast<float>(
      declare_parameter<double>("fused_prediction_max_yaw_step_deg", 12.0));
    degraded_odom_timeout_sec_ = declare_parameter<double>("degraded_odom_timeout_sec", 3.0);
    recovery_search_cooldown_sec_ = declare_parameter<double>("recovery_search_cooldown_sec", 3.0);
    recovery_max_odom_xy_error_ = static_cast<float>(
      declare_parameter<double>("recovery_max_odom_xy_error", 1.0));
    recovery_max_odom_yaw_error_deg_ = static_cast<float>(
      declare_parameter<double>("recovery_max_odom_yaw_error_deg", 20.0));
    recovery_max_fitness_score_ = static_cast<float>(
      declare_parameter<double>("recovery_max_fitness_score", 0.14));
    recovery_verification_count_threshold_ =
      declare_parameter<int>("recovery_verification_count_threshold", 6);
    recovery_verification_max_fitness_score_ = static_cast<float>(
      declare_parameter<double>("recovery_verification_max_fitness_score", 0.25));
    recovery_verification_max_xy_spread_ = static_cast<float>(
      declare_parameter<double>("recovery_verification_max_xy_spread", 0.08));
    recovery_verification_max_yaw_spread_deg_ = static_cast<float>(
      declare_parameter<double>("recovery_verification_max_yaw_spread_deg", 3.0));
    recovery_stationary_linear_speed_threshold_ = static_cast<float>(
      declare_parameter<double>("recovery_stationary_linear_speed_threshold", 0.03));
    recovery_stationary_yaw_rate_threshold_ = static_cast<float>(
      declare_parameter<double>("recovery_stationary_yaw_rate_threshold", 0.03));
    stationary_constraint_enabled_ =
      declare_parameter<bool>("stationary_constraint_enabled", true);
    stationary_gyro_rate_threshold_ = static_cast<float>(
      declare_parameter<double>("stationary_gyro_rate_threshold", 0.03));
    recovery_stationary_max_xy_error_ = static_cast<float>(
      declare_parameter<double>("recovery_stationary_max_xy_error", 0.20));
    recovery_stationary_max_yaw_error_deg_ = static_cast<float>(
      declare_parameter<double>("recovery_stationary_max_yaw_error_deg", 5.0));
    degraded_max_translation_ = static_cast<float>(
      declare_parameter<double>("degraded_max_translation", 0.60));
    degraded_max_yaw_deg_ = static_cast<float>(
      declare_parameter<double>("degraded_max_yaw_deg", 60.0));
    map_pose_gate_enabled_ = declare_parameter<bool>("map_pose_gate_enabled", true);
    map_pose_gate_topic_ = declare_parameter<std::string>("map_pose_gate_topic", "/map");
    map_pose_gate_occupied_threshold_ =
      declare_parameter<int>("map_pose_gate_occupied_threshold", 50);
    map_pose_gate_reject_unknown_ =
      declare_parameter<bool>("map_pose_gate_reject_unknown", true);
    map_pose_gate_footprint_radius_ = static_cast<float>(
      declare_parameter<double>("map_pose_gate_footprint_radius", 0.20));
    map_pose_gate_segment_step_ = static_cast<float>(
      declare_parameter<double>("map_pose_gate_segment_step", 0.05));
    deskew_enabled_ = declare_parameter<bool>("deskew_enabled", false);
    deskew_timestamp_field_ = declare_parameter<std::string>(
      "deskew_timestamp_field", "timestamp");
    deskew_timestamp_scale_sec_ = declare_parameter<double>(
      "deskew_timestamp_scale_sec", 1e-9);
    deskew_min_coverage_ = declare_parameter<double>(
      "deskew_min_coverage", 0.90);
    
    // Global localization parameters
    declare_parameter<bool>("use_global_localization_init", true);
    get_parameter("use_global_localization_init", use_global_localization_init_);
    
    declare_parameter<float>("init_pose_change_threshold", 0.01);
    get_parameter("init_pose_change_threshold", init_pose_change_threshold_);
    
    declare_parameter<float>("init_quat_change_threshold", 0.01);
    get_parameter("init_quat_change_threshold", init_quat_change_threshold_);
    
    declare_parameter<float>("global_localization_timeout", 10.0);
    get_parameter("global_localization_timeout", global_localization_timeout_);

    static_imu_init_.SetParam(imu_init_time_, imu_init_queue_size_, imu_init_max_gyro_var_, imu_init_max_acce_var_);
    
    if (!init_imu_R.empty()) {
      init_rotation_matrix_ << init_imu_R[0], init_imu_R[1], init_imu_R[2], 
                               init_imu_R[3], init_imu_R[4], init_imu_R[5], 
                               init_imu_R[6], init_imu_R[7], init_imu_R[8];
      RCLCPP_INFO(get_logger(), "IMU rotation matrix loaded from config");
    }
    if (!init_T.empty()) {
      gravity_transform_ << init_T[0], init_T[1], init_T[2], init_T[3], 
                           init_T[4], init_T[5], init_T[6], init_T[7], 
                           init_T[8], init_T[9], init_T[10], init_T[11], 
                           init_T[12], init_T[13], init_T[14], init_T[15];
      RCLCPP_INFO(get_logger(), "Gravity transform matrix loaded from config");
    }
    // Log loaded parameters for verification
    RCLCPP_INFO(get_logger(),
                "Loaded parameters:\n"
                "  robot_odom_frame_id: %s\n"
                "  odom_child_frame_id: %s\n"
                "  localization_odom_frame_id: %s\n"
                "  use_imu: %s\n"
                "  invert_acc: %s\n"
                "  invert_gyro: %s\n"
                "  imu_topic: %s\n"
                "  points_topic: %s\n"
                "  odom_topic: %s\n"
                "  aligned_points_topic: %s\n"
                "  status_topic: %s\n"
                "  localization_info_topic: %s\n"
                "  localization_health_topic: %s\n"
                "  global_map_points_topic: %s\n"
                "  send_tf_transforms: %s\n"
                "  enable_robot_odometry_prediction: %s\n"
                "  reg_method: %s\n"
                "  ndt_neighbor_search_method: %s\n"
                "  ndt_neighbor_search_radius: %.3f\n"
                "  ndt_resolution: %.3f\n"
                "  imu_data_filter_num: %d\n"
                "  globalmap_voxel_size: %.3f\n"
                "  points_voxel_filter_size: %.3f\n"
                "  min_valid_count: %d\n"
                "  buffer_size: %d\n"
                "  imu_init_time: %.1f\n"
                "  imu_init_queue_size: %d\n"
                "  imu_init_max_gyro_var: %.3f\n"
                "  imu_init_max_acce_var: %.3f",
                robot_odom_frame_id.c_str(),
                odom_child_frame_id.c_str(),
                localization_odom_frame_id.c_str(),
                use_imu ? "true" : "false",
                invert_acc ? "true" : "false",
                invert_gyro ? "true" : "false",
                imu_topic.c_str(),
                points_topic.c_str(),
                odom_topic.c_str(),
                aligned_points_topic.c_str(),
                status_topic.c_str(),
                localization_info_topic.c_str(),
                localization_health_topic.c_str(),
                global_map_points_topic.c_str(),
                send_tf_transforms ? "true" : "false",
                enable_robot_odometry_prediction ? "true" : "false",
                reg_method.c_str(),
                ndt_neighbor_search_method.c_str(),
                ndt_neighbor_search_radius,
                ndt_resolution,
                imu_data_filter_num_,
                globalmap_voxel_size_,
                points_voxel_filter_size_,
                min_valid_count_,
                buffer_size_,
                imu_init_time_,
                imu_init_queue_size_,
                imu_init_max_gyro_var_,
                imu_init_max_acce_var_);

    global_map_points_ptr_.reset(new pcl::PointCloud<PointT>());


    if (use_imu) {
      auto imu_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
      rclcpp::SubscriptionOptions imu_options;
      imu_options.callback_group = imu_group;
      RCLCPP_INFO(get_logger(), "enable imu-based prediction");
      correct_imu_data_ptr_ = std::make_shared<sensor_msgs::msg::Imu>();
      imu_sub               = create_subscription<sensor_msgs::msg::Imu>(imu_topic, 128, std::bind(&HdlLocalizationNode::imu_callback, this, std::placeholders::_1), imu_options);
    }

    if (enable_robot_odometry_prediction) {
      auto motor_odom_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
      rclcpp::SubscriptionOptions motor_odom_options;
      motor_odom_options.callback_group = motor_odom_group;
      motor_odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        motor_odom_topic, rclcpp::QoS(rclcpp::KeepLast(64)).best_effort(),
        std::bind(&HdlLocalizationNode::motor_odom_callback, this, std::placeholders::_1),
        motor_odom_options);
      RCLCPP_INFO(get_logger(),
        "Fused predictor enabled: translation from %s twist, yaw from IMU gyro",
        motor_odom_topic.c_str());
      RCLCPP_INFO(
        get_logger(), "Internal odometry UKF is %s; deterministic fused prediction remains active",
        enable_internal_odom_ukf_ ? "enabled" : "disabled");
    }

    auto points_qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    points_sub      = create_subscription<sensor_msgs::msg::PointCloud2>(points_topic, points_qos, std::bind(&HdlLocalizationNode::points_callback, this, std::placeholders::_1));
    initialpose_sub =
      create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>("/initialpose", 8, std::bind(&HdlLocalizationNode::initialpose_callback, this, std::placeholders::_1));
    if (map_pose_gate_enabled_) {
      auto map_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
      occupancy_map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        map_pose_gate_topic_, map_qos,
        std::bind(&HdlLocalizationNode::occupancy_map_callback, this, std::placeholders::_1));
    }

    localization_lidar_info_timer_ = this->create_wall_timer(
                std::chrono::milliseconds(100), // 10Hz 
                std::bind(&HdlLocalizationNode::PublishLidarLocalizationInfo, this));
    odom_publish_timer_            = this->create_wall_timer(
                std::chrono::milliseconds(50), // 20Hz
                std::bind(&HdlLocalizationNode::PublishOdomTimer, this));
    pose_pub    = create_publisher<nav_msgs::msg::Odometry>(odom_topic, 5);
    local_odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(local_odom_topic, 20);
    local_odom_publish_timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(1.0 / local_odom_publish_rate_hz_)),
      std::bind(&HdlLocalizationNode::PublishLocalOdomTimer, this));
    aligned_pub = create_publisher<sensor_msgs::msg::PointCloud2>(aligned_points_topic, 5);
    status_pub  = create_publisher<localization::msg::ScanMatchingStatus>(status_topic, 5);

    localization_state_srv_ = this->create_service<robots_dog_msgs::srv::LocalizationState>(
            "/localization_state/service",
            std::bind(&HdlLocalizationNode::LocalizationStateCallback, this, std::placeholders::_1, std::placeholders::_2));
    load_map_service_ptr_   = create_service<robots_dog_msgs::srv::LoadMap>(
            "/load_map_service", std::bind(&HdlLocalizationNode::LoadMapCallBack, this, std::placeholders::_1, std::placeholders::_2));

    localization_info_pub_ = this->create_publisher<robots_dog_msgs::msg::Localization>(localization_info_topic, 10);
    localization_health_pub_ =
      this->create_publisher<robots_dog_msgs::msg::Localization>(localization_health_topic, 10);
    global_map_pub_        = this->create_publisher<sensor_msgs::msg::PointCloud2>(global_map_points_topic, 1);

    initialize_params();
    raw_points_ptr_ = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>());
    // Initialize sensor data validity tracking
    last_lidar_data_time_ = get_clock()->now();
    last_imu_data_time_   = get_clock()->now();
    // Initialize buffer, mark all as invalid initially
    for (int i = 0; i < buffer_size_; i++) {
      lidar_status_buffer_.push_back(false);
      imu_status_buffer_.push_back(false);
    }
    // Initialize pose history
    last_valid_pose_time_   = get_clock()->now();
    has_valid_pose_history_ = false;
    
    // Initialize confidence management
    last_confidence_update_time_ = get_clock()->now();
    current_confidence_          = 0.0;
    is_extrapolating_            = false;
    
    log_counter_  = 0;
    log_interval_ = 10;
    
    RCLCPP_INFO(get_logger(), "Sensor data validity tracking initialized with buffer size: %d, min valid count: %d", 
                buffer_size_, min_valid_count_);
  }

  ~HdlLocalizationNode() override {
    if (recovery_future_.valid()) recovery_future_.wait();
  }

private:
  localization::StaticIMUInit static_imu_init_;
  bool imu_bias_applied_to_estimator_ = false;
  bool has_cached_imu_biases_ = false;
  Eigen::Vector3f cached_acc_bias_ = Eigen::Vector3f::Zero();
  Eigen::Vector3f cached_gyro_bias_ = Eigen::Vector3f::Zero();
  float stationary_reference_z_ = 0.0f;
  float stationary_reference_roll_ = 0.0f;
  float stationary_reference_pitch_ = 0.0f;
  float imu_init_time_ = 3.0f;
  int imu_init_queue_size_ = 300;
  float imu_init_max_gyro_var_ = 0.05f;
  float imu_init_max_acce_var_ = 0.2f;
  // Initial pose initialization parameters
  int init_match_count_threshold_ = 5;
  int bad__match_count_threshold_ = 6;
  int tracking_degraded_count_threshold_ = 2;
  float init_match_score_threshold_ = 0.2f;
  float reliable_threshold_ = 0.25f;
  float tracking_max_xy_jump_ = 0.3f;
  float tracking_max_yaw_jump_deg_ = 15.0f;
  bool planar_ndt_enabled_ = true;
  double localization_scan_min_interval_sec_ = 0.18;
  double motor_twist_stale_timeout_sec_ = 0.35;
  float motor_twist_filter_alpha_ = 0.25f;
  float fused_prediction_translation_margin_ = 0.05f;
  float fused_prediction_max_speed_ = 0.75f;
  float fused_prediction_max_yaw_step_deg_ = 12.0f;
  int64_t last_processed_scan_stamp_ns_ = 0;
  int64_t last_odom_prediction_stamp_ns_ = 0;
  // Initial pose initialization state variables
  int init_match_count_ = 0;
  int bad_match_count_ = 0;
  int recovery_count_multiplier_ = 1;
  rclcpp::Time last_recovery_search_time_{0, 0, RCL_ROS_TIME};
  double recovery_search_cooldown_sec_ = 3.0;
  double degraded_odom_timeout_sec_ = 3.0;
  float recovery_max_odom_xy_error_ = 1.0f;
  float recovery_max_odom_yaw_error_deg_ = 20.0f;
  float recovery_max_fitness_score_ = 0.14f;
  int recovery_verification_count_threshold_ = 6;
  float recovery_verification_max_fitness_score_ = 0.25f;
  float recovery_verification_max_xy_spread_ = 0.08f;
  float recovery_verification_max_yaw_spread_deg_ = 3.0f;
  float recovery_stationary_linear_speed_threshold_ = 0.03f;
  float recovery_stationary_yaw_rate_threshold_ = 0.03f;
  bool stationary_constraint_enabled_ = true;
  float stationary_gyro_rate_threshold_ = 0.03f;
  float recovery_stationary_max_xy_error_ = 0.20f;
  float recovery_stationary_max_yaw_error_deg_ = 5.0f;
  float degraded_max_translation_ = 0.60f;
  float degraded_max_yaw_deg_ = 60.0f;
  bool map_pose_gate_enabled_ = true;
  std::string map_pose_gate_topic_ = "/map";
  int map_pose_gate_occupied_threshold_ = 50;
  bool map_pose_gate_reject_unknown_ = true;
  float map_pose_gate_footprint_radius_ = 0.20f;
  float map_pose_gate_segment_step_ = 0.05f;
  bool deskew_enabled_ = false;
  std::string deskew_timestamp_field_ = "timestamp";
  double deskew_timestamp_scale_sec_ = 1e-9;
  double deskew_min_coverage_ = 0.90;
  uint64_t deskew_applied_scan_count_ = 0;
  uint64_t deskew_skipped_scan_count_ = 0;

  struct RecoveryResult {
    bool found = false;
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f source_odom_pose = Eigen::Matrix4f::Identity();
    int64_t source_stamp_ns = 0;
    double score = std::numeric_limits<double>::infinity();
    double best_observed_score = std::numeric_limits<double>::infinity();
    float best_observed_xy_error = std::numeric_limits<float>::infinity();
    float best_observed_yaw_error_deg = std::numeric_limits<float>::infinity();
  };

  struct FusedPredictionState {
    int64_t stamp_ns = 0;
    float x = 0.0f;
    float y = 0.0f;
    float yaw = 0.0f;
  };

  struct LocalizationOutput {
    bool available = false;
    uint8_t status = 0;
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    Eigen::Vector3f velocity = Eigen::Vector3f::Zero();
  };

  uint8_t current_localization_status_locked() const {
    if (degraded_odom_active_ || is_extrapolating_) return 4;
    if (localization_state_ < 0 || localization_state_ > 4) return 4;
    return static_cast<uint8_t>(localization_state_);
  }

  LocalizationOutput current_output_locked() const {
    LocalizationOutput output;
    output.status = current_localization_status_locked();

    if (degraded_odom_active_ && has_reliable_pose_) {
      output.pose = degraded_pose_;
      output.available = true;
      return output;
    }
    if (is_extrapolating_ && has_valid_pose_history_) {
      output.pose = last_pose_;
      output.velocity = last_velocity_;
      output.available = true;
      return output;
    }
    if (!is_init_success_ && has_initialization_hold_pose_) {
      output.pose = initialization_hold_pose_;
      output.available = true;
      return output;
    }
    if (pose_estimator) {
      output.pose = pose_estimator->matrix();
      if (is_init_success_) output.velocity = pose_estimator->vel();
      output.available = true;
      return output;
    }
    if (has_reliable_pose_) {
      output.pose = last_reliable_pose_;
      output.available = true;
    }
    return output;
  }

  std::array<double, 36> output_pose_covariance_locked(
    uint8_t status, bool pose_available = true) const {
    std::array<double, 36> covariance{};
    if (!pose_available) {
      covariance[0] = covariance[7] = covariance[14] = 1e6;
      covariance[21] = covariance[28] = covariance[35] = 1e6;
      return covariance;
    }
    if (status == 3 && pose_estimator && !degraded_odom_active_ && !is_extrapolating_) {
      const Eigen::Matrix<float, 6, 6> filter_covariance =
        pose_estimator->pose_covariance();
      for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 6; ++col) {
          covariance[static_cast<size_t>(row * 6 + col)] =
            static_cast<double>(filter_covariance(row, col));
        }
      }
      const double position_floor =
        tracking_min_position_stddev_ * tracking_min_position_stddev_;
      const double orientation_stddev =
        tracking_min_orientation_stddev_deg_ * M_PI / 180.0;
      const double orientation_floor = orientation_stddev * orientation_stddev;
      for (int axis = 0; axis < 3; ++axis) {
        covariance[static_cast<size_t>(axis * 6 + axis)] = std::max(
          covariance[static_cast<size_t>(axis * 6 + axis)], position_floor);
        covariance[static_cast<size_t>((axis + 3) * 6 + axis + 3)] = std::max(
          covariance[static_cast<size_t>((axis + 3) * 6 + axis + 3)],
          orientation_floor);
      }
      return covariance;
    }

    const bool degraded = status == 4;
    const double position_stddev = degraded ? degraded_position_stddev_ :
                                              initializing_position_stddev_;
    const double orientation_stddev_deg = degraded ?
      degraded_orientation_stddev_deg_ : initializing_orientation_stddev_deg_;
    const double position_variance = position_stddev * position_stddev;
    const double orientation_stddev = orientation_stddev_deg * M_PI / 180.0;
    const double orientation_variance = orientation_stddev * orientation_stddev;
    covariance[0] = covariance[7] = position_variance;
    covariance[14] = std::max(position_variance, 1.0);
    covariance[21] = covariance[28] = orientation_variance;
    covariance[35] = orientation_variance;
    return covariance;
  }

  std::mutex fused_prediction_mutex_;
  std::deque<FusedPredictionState> fused_prediction_history_;
  FusedPredictionState fused_prediction_state_;
  std::mutex map_to_odom_mutex_;
  Eigen::Matrix4f latest_map_to_odom_{Eigen::Matrix4f::Identity()};
  bool has_map_to_odom_ = false;
  int64_t latest_motor_twist_stamp_ns_ = 0;
  int64_t last_motor_odom_stamp_ns_ = 0;
  uint64_t motor_odom_received_count_ = 0;
  uint64_t motor_odom_duplicate_count_ = 0;
  uint64_t motor_odom_out_of_order_count_ = 0;
  float latest_motor_vx_ = 0.0f;
  float latest_motor_vy_ = 0.0f;
  bool has_motor_twist_ = false;
  bool has_motor_odom_frame_alignment_ = false;
  float motor_odom_to_fused_yaw_ = 0.0f;
  bool publish_vendor_localization_info_ = false;

  bool degraded_odom_active_ = false;
  bool degraded_odom_blocked_ = false;
  bool degraded_odom_stationary_episode_ = false;
  rclcpp::Time degraded_odom_start_time_{0, 0, RCL_ROS_TIME};
  Eigen::Matrix4f degraded_anchor_map_pose_{Eigen::Matrix4f::Identity()};
  Eigen::Matrix4f degraded_anchor_odom_pose_{Eigen::Matrix4f::Identity()};
  Eigen::Matrix4f degraded_pose_{Eigen::Matrix4f::Identity()};
  Eigen::Matrix4f last_reliable_odom_pose_{Eigen::Matrix4f::Identity()};
  bool has_reliable_odom_pose_ = false;
  int64_t last_reliable_scan_stamp_ns_ = 0;
  bool has_initialization_hold_pose_ = false;
  Eigen::Matrix4f initialization_hold_pose_{Eigen::Matrix4f::Identity()};
  bool recovery_verification_active_ = false;
  bool has_recovery_verification_anchor_ = false;
  Eigen::Matrix4f recovery_verification_anchor_map_pose_{Eigen::Matrix4f::Identity()};
  Eigen::Matrix4f recovery_verification_anchor_odom_pose_{Eigen::Matrix4f::Identity()};
  bool has_recovery_verification_offset_ = false;
  Eigen::Matrix4f recovery_verification_offset_{Eigen::Matrix4f::Identity()};
  std::mutex occupancy_map_mutex_;
  nav_msgs::msg::OccupancyGrid::ConstSharedPtr occupancy_map_;
  std::future<RecoveryResult> recovery_future_;

  void occupancy_map_callback(const nav_msgs::msg::OccupancyGrid::ConstSharedPtr msg) {
    std::lock_guard<std::mutex> lock(occupancy_map_mutex_);
    occupancy_map_ = msg;
  }

  bool point_is_known_free(const nav_msgs::msg::OccupancyGrid& map,
                           float world_x, float world_y) const {
    const auto& origin = map.info.origin;
    const double qx = origin.orientation.x;
    const double qy = origin.orientation.y;
    const double qz = origin.orientation.z;
    const double qw = origin.orientation.w;
    const double yaw = std::atan2(2.0 * (qw * qz + qx * qy),
                                  1.0 - 2.0 * (qy * qy + qz * qz));
    const double dx = world_x - origin.position.x;
    const double dy = world_y - origin.position.y;
    const double local_x = std::cos(yaw) * dx + std::sin(yaw) * dy;
    const double local_y = -std::sin(yaw) * dx + std::cos(yaw) * dy;
    if (map.info.resolution <= 0.0) return false;
    const int gx = static_cast<int>(std::floor(local_x / map.info.resolution));
    const int gy = static_cast<int>(std::floor(local_y / map.info.resolution));
    if (gx < 0 || gy < 0 || gx >= static_cast<int>(map.info.width) ||
        gy >= static_cast<int>(map.info.height)) return false;
    const int8_t value = map.data[static_cast<size_t>(gy) * map.info.width + gx];
    if (value < 0) return !map_pose_gate_reject_unknown_;
    return value < map_pose_gate_occupied_threshold_;
  }

  bool pose_is_map_safe(const Eigen::Matrix4f& pose) {
    if (!map_pose_gate_enabled_) return true;
    nav_msgs::msg::OccupancyGrid::ConstSharedPtr map;
    {
      std::lock_guard<std::mutex> lock(occupancy_map_mutex_);
      map = occupancy_map_;
    }
    if (!map) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "Map pose gate is enabled but no occupancy grid has been received; gate is temporarily open");
      return true;
    }
    const float x = pose(0, 3);
    const float y = pose(1, 3);
    const float r = map_pose_gate_footprint_radius_;
    static constexpr float kDiag = 0.70710678f;
    const float offsets[][2] = {
      {0.0f, 0.0f}, {r, 0.0f}, {-r, 0.0f}, {0.0f, r}, {0.0f, -r},
      {r * kDiag, r * kDiag}, {r * kDiag, -r * kDiag},
      {-r * kDiag, r * kDiag}, {-r * kDiag, -r * kDiag}
    };
    for (const auto& offset : offsets) {
      if (!point_is_known_free(*map, x + offset[0], y + offset[1])) return false;
    }
    return true;
  }

  bool transition_is_map_safe(const Eigen::Matrix4f& from,
                              const Eigen::Matrix4f& to) {
    const float distance = (to.block<2, 1>(0, 3) - from.block<2, 1>(0, 3)).norm();
    const float step = std::max(0.01f, map_pose_gate_segment_step_);
    const int samples = std::max(1, static_cast<int>(std::ceil(distance / step)));
    for (int i = 1; i <= samples; ++i) {
      const float alpha = static_cast<float>(i) / static_cast<float>(samples);
      Eigen::Matrix4f sample = from;
      sample.block<3, 1>(0, 3) =
        (1.0f - alpha) * from.block<3, 1>(0, 3) + alpha * to.block<3, 1>(0, 3);
      if (!pose_is_map_safe(sample)) return false;
    }
    return true;
  }

  bool odom_trajectory_is_map_safe(
      const Eigen::Matrix4f& candidate_pose, int64_t candidate_stamp_ns) {
    if (!has_reliable_pose_ || !has_reliable_odom_pose_ ||
        last_reliable_scan_stamp_ns_ == 0 ||
        candidate_stamp_ns <= last_reliable_scan_stamp_ns_) {
      return transition_is_map_safe(last_reliable_pose_, candidate_pose);
    }

    std::vector<FusedPredictionState> history;
    {
      std::lock_guard<std::mutex> lock(fused_prediction_mutex_);
      for (const auto& state : fused_prediction_history_) {
        if (state.stamp_ns >= last_reliable_scan_stamp_ns_ &&
            state.stamp_ns <= candidate_stamp_ns) {
          history.push_back(state);
        }
      }
    }
    if (history.empty()) {
      return transition_is_map_safe(last_reliable_pose_, candidate_pose);
    }

    const Eigen::Matrix4f map_from_odom =
      last_reliable_pose_ * last_reliable_odom_pose_.inverse();
    Eigen::Matrix4f previous = last_reliable_pose_;
    for (const auto& state : history) {
      const Eigen::Matrix4f sample = map_from_odom * fused_pose_matrix(state);
      if (!transition_is_map_safe(previous, sample)) return false;
      previous = sample;
    }
    return transition_is_map_safe(previous, candidate_pose);
  }

  void set_initial_correction_limits() {
    if (pose_estimator) {
      pose_estimator->set_planar_correction(planar_ndt_enabled_);
      pose_estimator->set_correction_limits(
        5.0f,
        std::numeric_limits<float>::infinity(),
        init_match_score_threshold_);
    }
  }

  void set_recovery_verification_correction_limits() {
    if (pose_estimator) {
      pose_estimator->set_planar_correction(planar_ndt_enabled_);
      pose_estimator->set_correction_limits(
        5.0f,
        std::numeric_limits<float>::infinity(),
        recovery_verification_max_fitness_score_);
    }
  }

  pcl::Registration<PointT, PointT>::Ptr create_registration(int num_threads) {
    if (reg_method == "NDT_OMP") {
      RCLCPP_INFO(get_logger(), "NDT_OMP is selected");
      pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
      ndt->setNumThreads(num_threads);
      ndt->setTransformationEpsilon(0.01);
      ndt->setResolution(ndt_resolution);
      if (ndt_neighbor_search_method == "DIRECT1") {
        RCLCPP_INFO(get_logger(), "search_method DIRECT1 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
      } else if (ndt_neighbor_search_method == "DIRECT7") {
        RCLCPP_INFO(get_logger(), "search_method DIRECT7 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
      } else {
        if (ndt_neighbor_search_method == "KDTREE") {
          RCLCPP_INFO(get_logger(), "search_method KDTREE is selected");
        } else {
          RCLCPP_WARN(get_logger(), "invalid search method was given");
          RCLCPP_WARN(get_logger(), "default method is selected (KDTREE)");
        }
        ndt->setNeighborhoodSearchMethod(pclomp::KDTREE);
      }
      return ndt;
    }
    RCLCPP_ERROR_STREAM(get_logger(), "unknown registration method:" << reg_method);
    return nullptr;
  }

  void initialize_params() {
    voxel_filter_ptr_->setLeafSize(points_voxel_filter_size_, points_voxel_filter_size_, points_voxel_filter_size_);
    registration = create_registration(6);

    // Initialize global localization
    if (use_global_localization_init_) {
      try {
        global_localization_ptr_ = std::make_shared<GlobalLocalization>();
        RCLCPP_INFO(get_logger(), "Global localization initialized successfully");
      } catch (const std::exception& e) {
        RCLCPP_WARN(get_logger(), "Failed to initialize global localization: %s", e.what());
        use_global_localization_init_ = false;
      }
    }
    // initialize pose estimator
    specify_init_pose_ = declare_parameter<bool>("specify_init_pose", true);
    if (specify_init_pose_) {
      RCLCPP_INFO(get_logger(), "initialize pose estimator with specified parameters!!"); 
      init_pos_x_ = declare_parameter<double>("init_pos_x", 0.0);
      init_pos_y_ = declare_parameter<double>("init_pos_y", 0.0);
      init_pos_z_ = declare_parameter<double>("init_pos_z", 0.0);
      init_ori_w_ = declare_parameter<double>("init_ori_w", 1.0);
      init_ori_x_ = declare_parameter<double>("init_ori_x", 0.0);
      init_ori_y_ = declare_parameter<double>("init_ori_y", 0.0);
      init_ori_z_ = declare_parameter<double>("init_ori_z", 0.0);
  
      Eigen::Vector3f config_pos(init_pos_x_, init_pos_y_, init_pos_z_);
      Eigen::Quaternionf config_quat(init_ori_w_, init_ori_x_, init_ori_y_, init_ori_z_);
      last_init_pos_ = config_pos;
      last_init_quat_ = config_quat;
      has_set_init_pose_ = true;
      last_pose_source_ = "config";
      pose_estimator.reset(new localization::PoseEstimator(
        registration,
        get_clock()->now(),
        last_init_pos_,
        last_init_quat_,
        cool_time_duration
      ));
      set_initial_correction_limits();
      apply_static_imu_biases_to_estimator();
      last_odom_prediction_stamp_ns_ = 0;
      RCLCPP_INFO(get_logger(), "Initial pose estimator created with config pose - Position: [%.3f, %.3f, %.3f]",
                   last_init_pos_.x(), last_init_pos_.y(), last_init_pos_.z());
    }
  }

private:
  void apply_static_imu_biases_to_estimator() {
    if (!use_imu || !pose_estimator || imu_bias_applied_to_estimator_) {
      return;
    }

    const float acc_sign = invert_acc ? -1.0f : 1.0f;
    const float gyro_sign = invert_gyro ? -1.0f : 1.0f;
    Eigen::Vector3f acc_bias;
    Eigen::Vector3f gyro_bias;
    {
      std::lock_guard<std::mutex> imu_lock(imu_data_mutex);
      if (!static_imu_init_.InitSuccess()) {
        return;
      }
      if (!has_cached_imu_biases_) {
        const Eigen::Vector3f measured_acc =
          acc_sign * static_imu_init_.GetInitBa().cast<float>();
        stationary_reference_z_ = last_init_pos_.z();
        const Eigen::Matrix3f reference_rotation = last_init_quat_.toRotationMatrix();
        stationary_reference_roll_ = std::atan2(
          reference_rotation(2, 1), reference_rotation(2, 2));
        stationary_reference_pitch_ = std::asin(std::clamp(
          -reference_rotation(2, 0), -1.0f, 1.0f));
        const Eigen::Quaternionf static_attitude(
          Eigen::AngleAxisf(stationary_reference_pitch_, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(stationary_reference_roll_, Eigen::Vector3f::UnitX()));
        const Eigen::Vector3f expected_body_gravity =
          static_attitude.inverse() * Eigen::Vector3f(0.0f, 0.0f, 9.80665f);
        cached_acc_bias_ = measured_acc - expected_body_gravity;
        cached_gyro_bias_ = gyro_sign * static_imu_init_.GetInitBg().cast<float>();
        has_cached_imu_biases_ = true;
        RCLCPP_INFO(
          get_logger(),
          "Cached planar reference: z=%.3f m roll=%.3f deg pitch=%.3f deg",
          stationary_reference_z_,
          stationary_reference_roll_ * 180.0f / static_cast<float>(M_PI),
          stationary_reference_pitch_ * 180.0f / static_cast<float>(M_PI));
      }
      acc_bias = cached_acc_bias_;
      gyro_bias = cached_gyro_bias_;
    }
    pose_estimator->set_initial_biases(acc_bias, gyro_bias);
    imu_bias_applied_to_estimator_ = true;
    RCLCPP_INFO(
      get_logger(),
      "Applied static IMU biases to UKF: acc=[%.5f %.5f %.5f] gyro=[%.6f %.6f %.6f]",
      acc_bias.x(), acc_bias.y(), acc_bias.z(),
      gyro_bias.x(), gyro_bias.y(), gyro_bias.z());
  }

  void mark_pose_estimator_recreated() {
    imu_bias_applied_to_estimator_ = false;
    apply_static_imu_biases_to_estimator();
  }

  void motor_odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr msg) {
    const int64_t stamp_ns = rclcpp::Time(msg->header.stamp).nanoseconds();
    std::lock_guard<std::mutex> lock(fused_prediction_mutex_);
    if (last_motor_odom_stamp_ns_ != 0 && stamp_ns <= last_motor_odom_stamp_ns_) {
      if (stamp_ns == last_motor_odom_stamp_ns_) {
        ++motor_odom_duplicate_count_;
      } else {
        ++motor_odom_out_of_order_count_;
      }
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "Ignoring non-monotonic motor odometry: stamp=%ld last=%ld duplicates=%llu out_of_order=%llu",
        static_cast<long>(stamp_ns), static_cast<long>(last_motor_odom_stamp_ns_),
        static_cast<unsigned long long>(motor_odom_duplicate_count_),
        static_cast<unsigned long long>(motor_odom_out_of_order_count_));
      return;
    }
    last_motor_odom_stamp_ns_ = stamp_ns;
    ++motor_odom_received_count_;
    const float vendor_vx = static_cast<float>(msg->twist.twist.linear.x);
    const float vendor_vy = static_cast<float>(msg->twist.twist.linear.y);
    if (!std::isfinite(vendor_vx) || !std::isfinite(vendor_vy)) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "Ignoring motor odometry with non-finite twist: vx=%f vy=%f",
        vendor_vx, vendor_vy);
      return;
    }
    // The vendor pose origin and all subsequent motor yaw samples are ignored.
    // Its first valid yaw is used once to identify the fixed orientation of
    // the vendor odom axes. Dynamic yaw remains exclusively IMU-driven.
    if (!has_motor_odom_frame_alignment_) {
      const auto& q = msg->pose.pose.orientation;
      const double norm_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
      if (!std::isfinite(norm_sq) || norm_sq < 1e-12) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Waiting for a valid initial motor odometry orientation; translation prediction held");
        return;
      }
      const double inv_norm = 1.0 / std::sqrt(norm_sq);
      const double x = q.x * inv_norm;
      const double y = q.y * inv_norm;
      const double z = q.z * inv_norm;
      const double w = q.w * inv_norm;
      const float motor_yaw = static_cast<float>(std::atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z)));
      motor_odom_to_fused_yaw_ = std::atan2(
        std::sin(fused_prediction_state_.yaw - motor_yaw),
        std::cos(fused_prediction_state_.yaw - motor_yaw));
      has_motor_odom_frame_alignment_ = true;
      RCLCPP_INFO(
        get_logger(),
        "Motor odom frame aligned once: motor_yaw=%.2f deg fused_yaw=%.2f deg offset=%.2f deg; future motor yaw ignored",
        motor_yaw * 180.0f / static_cast<float>(M_PI),
        fused_prediction_state_.yaw * 180.0f / static_cast<float>(M_PI),
        motor_odom_to_fused_yaw_ * 180.0f / static_cast<float>(M_PI));
    }

    // The vendor linear twist is expressed in its fixed odom/world axes.
    // Rotate it into the IMU-driven fused frame before integrating local deltas.
    latest_motor_twist_stamp_ns_ = stamp_ns;
    const float basis_cos = std::cos(motor_odom_to_fused_yaw_);
    const float basis_sin = std::sin(motor_odom_to_fused_yaw_);
    const float raw_vx = basis_cos * vendor_vx - basis_sin * vendor_vy;
    const float raw_vy = basis_sin * vendor_vx + basis_cos * vendor_vy;
    const float alpha = std::clamp(motor_twist_filter_alpha_, 0.0f, 1.0f);
    if (!has_motor_twist_) {
      latest_motor_vx_ = raw_vx;
      latest_motor_vy_ = raw_vy;
      has_motor_twist_ = true;
    } else {
      latest_motor_vx_ += alpha * (raw_vx - latest_motor_vx_);
      latest_motor_vy_ += alpha * (raw_vy - latest_motor_vy_);
    }
  }

  void update_fused_prediction(const rclcpp::Time& stamp, float imu_yaw_rate) {
    const int64_t stamp_ns = stamp.nanoseconds();
    std::lock_guard<std::mutex> lock(fused_prediction_mutex_);
    if (fused_prediction_state_.stamp_ns == 0) {
      fused_prediction_state_.stamp_ns = stamp_ns;
      fused_prediction_history_.push_back(fused_prediction_state_);
      return;
    }
    if (stamp_ns <= fused_prediction_state_.stamp_ns) return;

    const double dt = static_cast<double>(stamp_ns - fused_prediction_state_.stamp_ns) * 1e-9;
    if (dt > 0.05) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
        "Fused predictor IMU gap is %.3f s; translation held for this interval", dt);
    }
    const bool motor_twist_fresh = latest_motor_twist_stamp_ns_ != 0 &&
      std::abs(static_cast<double>(stamp_ns - latest_motor_twist_stamp_ns_) * 1e-9) <=
        motor_twist_stale_timeout_sec_;
    const float integration_dt = static_cast<float>(std::min(dt, 0.05));
    const float vx = motor_twist_fresh ? std::clamp(latest_motor_vx_, -1.5f, 1.5f) : 0.0f;
    const float vy = motor_twist_fresh ? std::clamp(latest_motor_vy_, -0.5f, 0.5f) : 0.0f;
    if (!std::isfinite(imu_yaw_rate)) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "Non-finite IMU yaw rate; holding yaw for this prediction step");
      imu_yaw_rate = 0.0f;
    }
    const float wz = std::clamp(imu_yaw_rate, -1.5f, 1.5f);
    // Motor twist has already been rotated from vendor odom axes into this
    // IMU-driven fused frame in motor_odom_callback().
    fused_prediction_state_.x += vx * integration_dt;
    fused_prediction_state_.y += vy * integration_dt;
    fused_prediction_state_.yaw = std::atan2(
      std::sin(fused_prediction_state_.yaw + wz * integration_dt),
      std::cos(fused_prediction_state_.yaw + wz * integration_dt));
    fused_prediction_state_.stamp_ns = stamp_ns;
    fused_prediction_history_.push_back(fused_prediction_state_);
    const int64_t oldest_allowed_ns = stamp_ns - static_cast<int64_t>(5e9);
    while (fused_prediction_history_.size() > 2 &&
           fused_prediction_history_[1].stamp_ns < oldest_allowed_ns) {
      fused_prediction_history_.pop_front();
    }
  }

  void imu_callback(const sensor_msgs::msg::Imu::SharedPtr imu_msg) {
    std::lock_guard<std::mutex> lock(imu_data_mutex);  // whole function - correct_imu_data_ptr_/latest_angular_velocity_ are read on other threads now
    const int64_t stamp_ns = rclcpp::Time(imu_msg->header.stamp).nanoseconds();
    ++imu_received_count_;
    if (last_received_imu_stamp_ns_ != 0 && stamp_ns <= last_received_imu_stamp_ns_) {
      if (stamp_ns == last_received_imu_stamp_ns_) {
        ++imu_duplicate_count_;
      } else {
        ++imu_out_of_order_count_;
      }
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "Ignoring non-monotonic IMU: stamp=%ld last=%ld duplicates=%llu out_of_order=%llu",
        static_cast<long>(stamp_ns), static_cast<long>(last_received_imu_stamp_ns_),
        static_cast<unsigned long long>(imu_duplicate_count_),
        static_cast<unsigned long long>(imu_out_of_order_count_));
      return;
    }
    last_received_imu_stamp_ns_ = stamp_ns;
    correct_imu_data_ptr_ = imu_msg;
    Eigen::Vector3f acceleration(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
    // Apply rotation matrix and gravity compensation
    acceleration = init_rotation_matrix_ * acceleration * 9.81;
    correct_imu_data_ptr_->linear_acceleration.x = acceleration.x();
    correct_imu_data_ptr_->linear_acceleration.y = acceleration.y();
    correct_imu_data_ptr_->linear_acceleration.z = acceleration.z();
    Eigen::Vector3f angular_velocity(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
    // Apply rotation matrix to angular velocity
    angular_velocity = init_rotation_matrix_ * angular_velocity;
    if (enable_robot_odometry_prediction) {
      const float gyro_sign = invert_gyro ? -1.0f : 1.0f;
      const float gyro_bias_z = has_cached_imu_biases_ ? cached_gyro_bias_.z() :
        (static_imu_init_.InitSuccess() ?
          gyro_sign * static_cast<float>(static_imu_init_.GetInitBg().z()) : 0.0f);
      update_fused_prediction(rclcpp::Time(imu_msg->header.stamp),
                              gyro_sign * angular_velocity.z() - gyro_bias_z);
    }
    correct_imu_data_ptr_->angular_velocity.x = angular_velocity.x();
    correct_imu_data_ptr_->angular_velocity.y = angular_velocity.y();
    correct_imu_data_ptr_->angular_velocity.z = angular_velocity.z();
    // Update latest IMU angular velocity for extrapolation
    latest_angular_velocity_ = angular_velocity;
    updateImuStatus(true);
    if (!static_imu_init_.InitSuccess()) {
      static_imu_init_.AddIMUData(correct_imu_data_ptr_);
    } else {
      imu_data.push_back(correct_imu_data_ptr_);  // already under imu_data_mutex above
      if (imu_data.size() > imu_buffer_max_size_) {
        const size_t overflow = imu_data.size() - imu_buffer_max_size_;
        imu_data.erase(imu_data.begin(), imu_data.begin() + overflow);
        imu_buffer_overflow_count_ += overflow;
        RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "IMU buffer overflow: dropped=%zu total_overflow=%llu",
          overflow, static_cast<unsigned long long>(imu_buffer_overflow_count_));
      }
    }
  }

  static float pose_yaw(const Eigen::Matrix4f& pose) {
    return std::atan2(pose(1, 0), pose(0, 0));
  }

  static Eigen::Matrix4f project_planar_pose(
      const Eigen::Matrix4f& candidate, float z, float roll, float pitch) {
    const float yaw = pose_yaw(candidate);
    const Eigen::Quaternionf attitude(
      Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) *
      Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *
      Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX()));

    Eigen::Matrix4f projected = candidate;
    projected.block<3, 3>(0, 0) = attitude.normalized().toRotationMatrix();
    projected(2, 3) = z;
    return projected;
  }

  static float angle_distance(float a, float b) {
    return std::abs(std::atan2(std::sin(a - b), std::cos(a - b)));
  }

  bool poses_agree(const Eigen::Matrix4f& lhs, const Eigen::Matrix4f& rhs,
                   float max_xy, float max_yaw_deg) const {
    const float xy = (lhs.block<2, 1>(0, 3) - rhs.block<2, 1>(0, 3)).norm();
    const float yaw = angle_distance(pose_yaw(lhs), pose_yaw(rhs));
    return xy <= max_xy && yaw <= max_yaw_deg * static_cast<float>(M_PI) / 180.0f;
  }

  bool lookup_odom_pose(const rclcpp::Time& stamp, Eigen::Matrix4f& pose) {
    const int64_t target_ns = stamp.nanoseconds();
    std::lock_guard<std::mutex> lock(fused_prediction_mutex_);
    if (fused_prediction_history_.empty()) return false;

    auto upper = std::lower_bound(
      fused_prediction_history_.begin(), fused_prediction_history_.end(), target_ns,
      [](const FusedPredictionState& state, int64_t value) {
        return state.stamp_ns < value;
      });
    FusedPredictionState state;
    if (upper == fused_prediction_history_.begin()) {
      state = *upper;
    } else if (upper == fused_prediction_history_.end()) {
      state = fused_prediction_history_.back();
      if (static_cast<double>(target_ns - state.stamp_ns) * 1e-9 >
          motor_twist_stale_timeout_sec_) return false;
    } else {
      const auto& before = *std::prev(upper);
      const auto& after = *upper;
      const double span = static_cast<double>(after.stamp_ns - before.stamp_ns);
      const float alpha = span > 0.0 ? static_cast<float>((target_ns - before.stamp_ns) / span) : 0.0f;
      state.x = before.x + alpha * (after.x - before.x);
      state.y = before.y + alpha * (after.y - before.y);
      const float yaw_delta = std::atan2(std::sin(after.yaw - before.yaw),
                                         std::cos(after.yaw - before.yaw));
      state.yaw = before.yaw + alpha * yaw_delta;
      state.stamp_ns = target_ns;
    }

    pose = Eigen::Matrix4f::Identity();
    pose(0, 0) = std::cos(state.yaw);
    pose(0, 1) = -std::sin(state.yaw);
    pose(1, 0) = std::sin(state.yaw);
    pose(1, 1) = std::cos(state.yaw);
    pose(0, 3) = state.x;
    pose(1, 3) = state.y;
    return true;
  }

  bool deterministic_fused_initial_guess(
      const rclcpp::Time& stamp, Eigen::Matrix4f& guess) {
    Eigen::Matrix4f current_odom_pose;
    if (!lookup_odom_pose(stamp, current_odom_pose)) return false;

    Eigen::Matrix4f anchor_map_pose;
    Eigen::Matrix4f anchor_odom_pose;
    if (recovery_verification_active_ && has_recovery_verification_anchor_) {
      anchor_map_pose = recovery_verification_anchor_map_pose_;
      anchor_odom_pose = recovery_verification_anchor_odom_pose_;
    } else if (has_reliable_pose_ && has_reliable_odom_pose_) {
      anchor_map_pose = last_reliable_pose_;
      anchor_odom_pose = last_reliable_odom_pose_;
    } else {
      return false;
    }

    guess = anchor_map_pose * anchor_odom_pose.inverse() * current_odom_pose;
    if (planar_ndt_enabled_) {
      guess = project_planar_pose(
        guess, anchor_map_pose(2, 3),
        stationary_reference_roll_, stationary_reference_pitch_);
    }
    return true;
  }

  static bool interpolate_fused_pose(
      const std::vector<FusedPredictionState>& history,
      int64_t target_ns,
      FusedPredictionState& state) {
    if (history.empty() || target_ns < history.front().stamp_ns ||
        target_ns > history.back().stamp_ns) {
      return false;
    }
    const auto upper = std::lower_bound(
      history.begin(), history.end(), target_ns,
      [](const FusedPredictionState& sample, int64_t value) {
        return sample.stamp_ns < value;
      });
    if (upper == history.begin()) {
      state = *upper;
      return true;
    }
    if (upper == history.end()) {
      state = history.back();
      return state.stamp_ns == target_ns;
    }
    const auto& before = *std::prev(upper);
    const auto& after = *upper;
    const double span = static_cast<double>(after.stamp_ns - before.stamp_ns);
    const float alpha = span > 0.0 ?
      static_cast<float>((target_ns - before.stamp_ns) / span) : 0.0f;
    state.x = before.x + alpha * (after.x - before.x);
    state.y = before.y + alpha * (after.y - before.y);
    const float yaw_delta = std::atan2(
      std::sin(after.yaw - before.yaw), std::cos(after.yaw - before.yaw));
    state.yaw = before.yaw + alpha * yaw_delta;
    state.stamp_ns = target_ns;
    return true;
  }

  static Eigen::Matrix4f fused_pose_matrix(const FusedPredictionState& state) {
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    pose(0, 0) = std::cos(state.yaw);
    pose(0, 1) = -std::sin(state.yaw);
    pose(1, 0) = std::sin(state.yaw);
    pose(1, 1) = std::cos(state.yaw);
    pose(0, 3) = state.x;
    pose(1, 3) = state.y;
    return pose;
  }

  static const sensor_msgs::msg::PointField* find_cloud_field(
      const sensor_msgs::msg::PointCloud2& cloud, const std::string& name) {
    const auto iter = std::find_if(
      cloud.fields.begin(), cloud.fields.end(),
      [&](const auto& field) { return field.name == name; });
    return iter == cloud.fields.end() ? nullptr : &*iter;
  }

  static bool read_point_time(
      const uint8_t* point,
      const sensor_msgs::msg::PointField& field,
      double scale_sec,
      double& offset_sec) {
    switch (field.datatype) {
      case sensor_msgs::msg::PointField::FLOAT64: {
        double value = 0.0;
        std::memcpy(&value, point + field.offset, sizeof(value));
        offset_sec = value * scale_sec;
        return std::isfinite(offset_sec);
      }
      case sensor_msgs::msg::PointField::FLOAT32: {
        float value = 0.0f;
        std::memcpy(&value, point + field.offset, sizeof(value));
        offset_sec = static_cast<double>(value) * scale_sec;
        return std::isfinite(offset_sec);
      }
      case sensor_msgs::msg::PointField::UINT32: {
        uint32_t value = 0;
        std::memcpy(&value, point + field.offset, sizeof(value));
        offset_sec = static_cast<double>(value) * scale_sec;
        return true;
      }
      default:
        return false;
    }
  }

  bool deskew_pointcloud(
      const sensor_msgs::msg::PointCloud2& input,
      sensor_msgs::msg::PointCloud2& output,
      double& scan_duration_sec,
      uint32_t& deskewed_points) {
    scan_duration_sec = 0.0;
    deskewed_points = 0;
    if (!deskew_enabled_ || input.is_bigendian) return false;
    const auto* x_field = find_cloud_field(input, "x");
    const auto* y_field = find_cloud_field(input, "y");
    const auto* z_field = find_cloud_field(input, "z");
    const auto* time_field = find_cloud_field(input, deskew_timestamp_field_);
    if (!x_field || !y_field || !z_field || !time_field ||
        x_field->datatype != sensor_msgs::msg::PointField::FLOAT32 ||
        y_field->datatype != sensor_msgs::msg::PointField::FLOAT32 ||
        z_field->datatype != sensor_msgs::msg::PointField::FLOAT32) {
      ++deskew_skipped_scan_count_;
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Deskew skipped: PointCloud2 lacks compatible x/y/z/%s fields",
        deskew_timestamp_field_.c_str());
      return false;
    }

    std::vector<FusedPredictionState> history;
    {
      std::lock_guard<std::mutex> lock(fused_prediction_mutex_);
      history.assign(fused_prediction_history_.begin(), fused_prediction_history_.end());
    }
    const int64_t scan_stamp_ns = rclcpp::Time(input.header.stamp).nanoseconds();
    FusedPredictionState reference_state;
    if (!interpolate_fused_pose(history, scan_stamp_ns, reference_state)) {
      ++deskew_skipped_scan_count_;
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Deskew skipped: fused odometry does not cover scan start");
      return false;
    }

    output = input;
    const Eigen::Matrix4f reference_pose = fused_pose_matrix(reference_state);
    const Eigen::Matrix4f lidar_to_base = gravity_transform_;
    const Eigen::Matrix4f base_to_lidar = lidar_to_base.inverse();
    uint32_t timestamped_points = 0;
    for (uint32_t row = 0; row < output.height; ++row) {
      for (uint32_t column = 0; column < output.width; ++column) {
        const size_t offset = static_cast<size_t>(row) * output.row_step +
          static_cast<size_t>(column) * output.point_step;
        if (offset + output.point_step > output.data.size()) continue;
        uint8_t* point = output.data.data() + offset;
        double point_offset_sec = 0.0;
        if (!read_point_time(
              point, *time_field, deskew_timestamp_scale_sec_, point_offset_sec) ||
            point_offset_sec < 0.0 || point_offset_sec > 0.5) {
          continue;
        }
        ++timestamped_points;
        scan_duration_sec = std::max(scan_duration_sec, point_offset_sec);
        FusedPredictionState point_state;
        const int64_t point_stamp_ns = scan_stamp_ns + static_cast<int64_t>(
          point_offset_sec * 1e9);
        if (!interpolate_fused_pose(history, point_stamp_ns, point_state)) continue;

        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        std::memcpy(&x, point + x_field->offset, sizeof(x));
        std::memcpy(&y, point + y_field->offset, sizeof(y));
        std::memcpy(&z, point + z_field->offset, sizeof(z));
        const Eigen::Matrix4f point_pose = fused_pose_matrix(point_state);
        const Eigen::Matrix4f point_to_reference =
          base_to_lidar * reference_pose.inverse() * point_pose * lidar_to_base;
        const Eigen::Vector4f corrected =
          point_to_reference * Eigen::Vector4f(x, y, z, 1.0f);
        const float corrected_x = corrected.x();
        const float corrected_y = corrected.y();
        const float corrected_z = corrected.z();
        std::memcpy(point + x_field->offset, &corrected_x, sizeof(float));
        std::memcpy(point + y_field->offset, &corrected_y, sizeof(float));
        std::memcpy(point + z_field->offset, &corrected_z, sizeof(float));
        ++deskewed_points;
      }
    }
    const double coverage = timestamped_points > 0 ?
      static_cast<double>(deskewed_points) / timestamped_points : 0.0;
    if (deskewed_points == 0 || coverage < deskew_min_coverage_) {
      ++deskew_skipped_scan_count_;
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Deskew skipped: fused odometry covers %.1f%% of timestamped points (minimum %.1f%%)",
        100.0 * coverage, 100.0 * deskew_min_coverage_);
      deskewed_points = 0;
      return false;
    }
    ++deskew_applied_scan_count_;
    return true;
  }

  bool robot_is_stationary_for_recovery(const rclcpp::Time& stamp) {
    float gyro_rate = std::numeric_limits<float>::infinity();
    {
      std::lock_guard<std::mutex> lock(imu_data_mutex);
      if (correct_imu_data_ptr_) {
        const float gyro_sign = invert_gyro ? -1.0f : 1.0f;
        Eigen::Vector3f corrected_gyro(
          correct_imu_data_ptr_->angular_velocity.x,
          correct_imu_data_ptr_->angular_velocity.y,
          correct_imu_data_ptr_->angular_velocity.z);
        corrected_gyro *= gyro_sign;
        if (has_cached_imu_biases_) {
          corrected_gyro -= cached_gyro_bias_;
        }
        gyro_rate = corrected_gyro.norm();
      }
    }

    float linear_speed = std::numeric_limits<float>::infinity();
    bool motor_twist_fresh = false;
    {
      std::lock_guard<std::mutex> lock(fused_prediction_mutex_);
      motor_twist_fresh = latest_motor_twist_stamp_ns_ != 0 &&
        std::abs(static_cast<double>(stamp.nanoseconds() - latest_motor_twist_stamp_ns_) * 1e-9) <=
          motor_twist_stale_timeout_sec_;
      if (motor_twist_fresh) {
        linear_speed = std::hypot(latest_motor_vx_, latest_motor_vy_);
      }
    }

    return motor_twist_fresh &&
      linear_speed <= recovery_stationary_linear_speed_threshold_ &&
      gyro_rate <= stationary_gyro_rate_threshold_;
  }

  bool recovery_pose_agrees_with_prediction(const Eigen::Matrix4f& pose,
                                             const rclcpp::Time& stamp,
                                             float* max_xy = nullptr,
                                             float* max_yaw_deg = nullptr) {
    // A navigation safety stop must not retroactively classify a loss that
    // happened in motion as a stationary recovery. Otherwise the strict
    // stationary gate can permanently reject the correction needed to remove
    // odometry drift accumulated before the stop.
    const bool stationary = degraded_odom_stationary_episode_ &&
      robot_is_stationary_for_recovery(stamp);
    const float xy_limit = stationary ? recovery_stationary_max_xy_error_ :
                                        recovery_max_odom_xy_error_;
    const float yaw_limit = stationary ? recovery_stationary_max_yaw_error_deg_ :
                                         recovery_max_odom_yaw_error_deg_;
    if (max_xy) *max_xy = xy_limit;
    if (max_yaw_deg) *max_yaw_deg = yaw_limit;
    return poses_agree(pose, degraded_pose_, xy_limit, yaw_limit);
  }

  void enter_degraded_odom(const rclcpp::Time& stamp) {
    if (degraded_odom_active_ || !has_reliable_pose_) return;
    degraded_odom_active_ = true;
    degraded_odom_blocked_ = false;
    degraded_odom_stationary_episode_ = robot_is_stationary_for_recovery(stamp);
    degraded_odom_start_time_ = stamp;
    degraded_anchor_map_pose_ = last_reliable_pose_;
    degraded_pose_ = last_reliable_pose_;
    if (has_reliable_odom_pose_) {
      degraded_anchor_odom_pose_ = last_reliable_odom_pose_;
    } else {
      has_reliable_odom_pose_ = lookup_odom_pose(stamp, degraded_anchor_odom_pose_);
    }
    current_confidence_ = 0.0;
    localization_state_ = 4;
    RCLCPP_WARN(get_logger(),
      "Entering DEGRADED_ODOM after rejected NDT correction (stationary_episode=%s)",
      degraded_odom_stationary_episode_ ? "true" : "false");
  }

  void update_degraded_odom(const rclcpp::Time& stamp) {
    if (!degraded_odom_active_ || !has_reliable_odom_pose_ || degraded_odom_blocked_) return;
    if (degraded_odom_stationary_episode_ &&
        !robot_is_stationary_for_recovery(stamp)) {
      degraded_odom_stationary_episode_ = false;
      RCLCPP_INFO(get_logger(),
        "DEGRADED_ODOM recovery switched to moving limits after motion was observed");
    }
    Eigen::Matrix4f current_odom;
    if (lookup_odom_pose(stamp, current_odom)) {
      const Eigen::Matrix4f odom_delta = degraded_anchor_odom_pose_.inverse() * current_odom;
      const float translation = odom_delta.block<2, 1>(0, 3).norm();
      const float yaw_deg = angle_distance(pose_yaw(odom_delta), 0.0f) * 180.0f /
                            static_cast<float>(M_PI);
      if (translation > degraded_max_translation_ || yaw_deg > degraded_max_yaw_deg_) {
        degraded_odom_blocked_ = true;
        localization_state_ = 4;
        RCLCPP_ERROR(get_logger(),
          "DEGRADED_ODOM frozen: motion budget exceeded (xy=%.3f/%.3f m, yaw=%.1f/%.1f deg)",
          translation, degraded_max_translation_, yaw_deg, degraded_max_yaw_deg_);
        return;
      }
      const Eigen::Matrix4f candidate = degraded_anchor_map_pose_ * odom_delta;
      if (!transition_is_map_safe(degraded_pose_, candidate)) {
        degraded_odom_blocked_ = true;
        localization_state_ = 4;
        RCLCPP_ERROR(get_logger(),
          "DEGRADED_ODOM frozen: motor-odometry path enters occupied or unknown map space");
        return;
      }
      degraded_pose_ = candidate;
    }
  }

  void start_recovery_search(const pcl::PointCloud<PointT>::Ptr& cloud,
                             const Eigen::Matrix4f& expected_pose,
                             const rclcpp::Time& source_stamp) {
    if (!degraded_odom_active_ || !global_localization_ptr_ ||
        !global_map_points_ptr_ || global_map_points_ptr_->empty()) return;
    if (recovery_verification_active_) return;
    if ((get_clock()->now() - degraded_odom_start_time_).seconds() < degraded_odom_timeout_sec_) return;
    // A ready result must be consumed by the scan callback before another
    // search starts; otherwise a valid recovery could be silently discarded.
    if (recovery_future_.valid()) return;
    if ((get_clock()->now() - last_recovery_search_time_).seconds() < recovery_search_cooldown_sec_) return;

    last_recovery_search_time_ = get_clock()->now();
    auto cloud_copy = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>(*cloud));
    auto map = global_map_points_ptr_;
    const float max_xy = recovery_max_odom_xy_error_;
    const float max_yaw = recovery_max_odom_yaw_error_deg_;
    const float max_score = recovery_max_fitness_score_;
    const float planar_z = stationary_reference_z_;
    const float planar_roll = stationary_reference_roll_;
    const float planar_pitch = stationary_reference_pitch_;
    Eigen::Matrix4f source_odom_pose;
    if (!lookup_odom_pose(source_stamp, source_odom_pose)) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "Recovery search skipped: fused pose is unavailable at source scan");
      return;
    }
    const int64_t source_stamp_ns = source_stamp.nanoseconds();

    recovery_future_ = std::async(std::launch::async,
      [this, cloud_copy, map, expected_pose, max_xy, max_yaw, max_score,
       planar_z, planar_roll, planar_pitch, source_odom_pose, source_stamp_ns]() {
        RecoveryResult result;
        result.source_odom_pose = source_odom_pose;
        result.source_stamp_ns = source_stamp_ns;
        auto recovery_registration = create_registration(4);
        if (!recovery_registration) return result;
        recovery_registration->setInputTarget(map);
        recovery_registration->setInputSource(cloud_copy);
        const auto candidates = global_localization_ptr_->generateCandidates(expected_pose.cast<double>());
        for (const auto& candidate : candidates) {
          pcl::PointCloud<PointT> aligned;
          recovery_registration->align(aligned, candidate.cast<float>());
          if (!recovery_registration->hasConverged()) continue;
          const double score = recovery_registration->getFitnessScore();
          const Eigen::Matrix4f pose = project_planar_pose(
            recovery_registration->getFinalTransformation(),
            planar_z, planar_roll, planar_pitch);
          const float xy_error =
            (pose.block<2, 1>(0, 3) - expected_pose.block<2, 1>(0, 3)).norm();
          const float yaw_error_deg = angle_distance(pose_yaw(pose), pose_yaw(expected_pose)) *
            180.0f / static_cast<float>(M_PI);
          if (score < result.best_observed_score) {
            result.best_observed_score = score;
            result.best_observed_xy_error = xy_error;
            result.best_observed_yaw_error_deg = yaw_error_deg;
          }
          if (score > max_score || !poses_agree(pose, expected_pose, max_xy, max_yaw) ||
              !pose_is_map_safe(pose) || !transition_is_map_safe(expected_pose, pose)) continue;
          if (!result.found || score < result.score) {
            result.found = true;
            result.pose = pose;
            result.score = score;
          }
        }
        return result;
      });
    RCLCPP_WARN(get_logger(), "Recovery NDT candidate search started in background");
  }

  bool consume_recovery_result(const rclcpp::Time& stamp) {
    if (!recovery_future_.valid() ||
        recovery_future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
    const RecoveryResult result = recovery_future_.get();
    if (!degraded_odom_active_) return false;
    Eigen::Matrix4f current_odom_pose;
    if (!lookup_odom_pose(stamp, current_odom_pose)) {
      RCLCPP_WARN(get_logger(),
        "Background recovery rejected: fused pose is unavailable at verification scan");
      return false;
    }
    const Eigen::Matrix4f propagated_pose =
      result.pose * result.source_odom_pose.inverse() * current_odom_pose;
    float max_xy = recovery_max_odom_xy_error_;
    float max_yaw_deg = recovery_max_odom_yaw_error_deg_;
    const bool agrees_with_prediction = result.found &&
      recovery_pose_agrees_with_prediction(propagated_pose, stamp, &max_xy, &max_yaw_deg);
    if (!agrees_with_prediction) {
      RCLCPP_WARN(get_logger(),
        "Background recovery rejected: best score=%.4f xy_error=%.3f/%.3f m yaw_error=%.1f/%.1f deg",
        result.best_observed_score, result.best_observed_xy_error,
        max_xy, result.best_observed_yaw_error_deg, max_yaw_deg);
      return false;
    }
    initialization_hold_pose_ = degraded_pose_;
    has_initialization_hold_pose_ = true;
    recovery_verification_anchor_map_pose_ = result.pose;
    recovery_verification_anchor_odom_pose_ = result.source_odom_pose;
    has_recovery_verification_anchor_ = true;
    is_init_success_ = false;
    init_match_count_ = 0;
    recovery_verification_active_ = true;
    has_recovery_verification_offset_ = false;
    RCLCPP_WARN(get_logger(),
      "Background recovery candidate accepted for stateless %d-scan verification, score=%.4f limits=%.3f m/%.1f deg",
      recovery_verification_count_threshold_, result.score, max_xy, max_yaw_deg);
    return true;
  }


  void points_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr points_msg) {
    updateLidarStatus(true);
    const int64_t scan_stamp_ns = rclcpp::Time(points_msg->header.stamp).nanoseconds();
    if (last_processed_scan_stamp_ns_ != 0) {
      const double scan_dt = static_cast<double>(scan_stamp_ns - last_processed_scan_stamp_ns_) * 1e-9;
      if (scan_dt < 0.0 || scan_dt < localization_scan_min_interval_sec_) {
        return;
      }
    }
    last_processed_scan_stamp_ns_ = scan_stamp_ns;

    auto start = std::chrono::high_resolution_clock::now();
    std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex); 
    if (use_imu && !static_imu_init_.InitSuccess()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5.0, "Radar CallBack Waiting for IMU Initial !!!");
      publishExtrapolatedOdom(points_msg->header.stamp);
      return;
    }
    if (!pose_estimator) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5.0, "Radar CallBack Waiting for Initial Pose Input!!");
      pubDefaultLocalizationOdom(points_msg->header.stamp);
      return;
    }
    apply_static_imu_biases_to_estimator();
    if (!global_map_points_ptr_ || global_map_points_ptr_->empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5.0, "Radar CallBack Waiting for Globalmap Input!!");
      pubDefaultLocalizationOdom(points_msg->header.stamp);
      return;
    }
    if (!isSensorDataValid()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1.0, "Sensor data invalid, using extrapolation!");
      publishExtrapolatedOdom(points_msg->header.stamp);
      return;
    }
  
    const auto& stamp = points_msg->header.stamp;
    pcl::PointCloud<PointT>::Ptr pcl_cloud(new pcl::PointCloud<PointT>());
    raw_points_ptr_->clear();
    sensor_msgs::msg::PointCloud2 deskewed_msg;
    double scan_duration_sec = 0.0;
    uint32_t deskewed_point_count = 0;
    const bool deskew_applied = deskew_pointcloud(
      *points_msg, deskewed_msg, scan_duration_sec, deskewed_point_count);
    pcl::fromROSMsg(
      deskew_applied ? deskewed_msg : *points_msg, *raw_points_ptr_);
    const size_t input_point_count = raw_points_ptr_->size();
                      
    if (raw_points_ptr_->empty()) {
      RCLCPP_ERROR(get_logger(), "cloud is empty!!");
      return;
    }

    auto filtered = downsample(raw_points_ptr_);
    TransformPoints(filtered, raw_points_ptr_);
    const size_t filtered_point_count = raw_points_ptr_->size();
    // last_scan = filtered;
    auto t_downsample = std::chrono::high_resolution_clock::now();

    if (use_global_localization_init_ && gl_once_gate_ && global_localization_ptr_ && !is_init_success_) {
      RCLCPP_INFO(get_logger(), "Attempting global localization for better initial pose...");
      if (performGlobalLocalization(raw_points_ptr_)) {
        RCLCPP_INFO(get_logger(), "Global localization successful! Using new initial pose.");
        pose_estimator.reset(new localization::PoseEstimator(
          registration, get_clock()->now(), last_init_pos_, last_init_quat_, cool_time_duration));
        set_initial_correction_limits();
        mark_pose_estimator_recreated();
        last_odom_prediction_stamp_ns_ = 0;
        is_init_success_ = false;
        init_match_count_ = 0;
        recovery_verification_active_ = false;
        has_recovery_verification_anchor_ = false;
        has_recovery_verification_offset_ = false;
        localization_state_ = 1;
        gl_once_gate_ = false;
        RCLCPP_INFO(get_logger(), "Pose estimator recreated with global localization result");
      } else {
        RCLCPP_INFO(get_logger(), "Global localization failed, continuing with current pose.");
      }
    }
    
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());  
    update_degraded_odom(rclcpp::Time(stamp));
    consume_recovery_result(rclcpp::Time(stamp));

    // predict
    if (!use_imu) {
      if (!recovery_verification_active_) {
        pose_estimator->predict(stamp);
      }
    } else {
      std::vector<sensor_msgs::msg::Imu::ConstSharedPtr> interval_imu;
      size_t future_queue_size = 0;
      {
        std::lock_guard<std::mutex> lock(imu_data_mutex);
        auto erase_end = imu_data.begin();
        for (auto iter = imu_data.begin(); iter != imu_data.end(); ++iter) {
          const int64_t imu_stamp_ns = rclcpp::Time((*iter)->header.stamp).nanoseconds();
          if (imu_stamp_ns > scan_stamp_ns) break;
          erase_end = std::next(iter);
          if (last_main_ukf_imu_stamp_ns_ != 0 &&
              imu_stamp_ns <= last_main_ukf_imu_stamp_ns_) {
            ++imu_main_ukf_rejected_count_;
            continue;
          }
          interval_imu.push_back(*iter);
        }
        imu_data.erase(imu_data.begin(), erase_end);
        future_queue_size = imu_data.size();
        imu_future_retained_count_ += future_queue_size;
      }

      const size_t stride = static_cast<size_t>(std::max(1, imu_data_filter_num_));
      for (size_t index = 0; index < interval_imu.size(); ++index) {
        const bool selected_by_stride = ((index + 1) % stride) == 0;
        const bool is_last_sample = index + 1 == interval_imu.size();
        if (!selected_by_stride && !is_last_sample) continue;

        const auto& imu = interval_imu[index];
        const auto& acc = imu->linear_acceleration;
        const auto& gyro = imu->angular_velocity;
        const double acc_sign = invert_acc ? -1.0 : 1.0;
        const double gyro_sign = invert_gyro ? -1.0 : 1.0;
        if (!recovery_verification_active_) {
          pose_estimator->predict(
            imu->header.stamp,
            acc_sign * Eigen::Vector3f(acc.x, acc.y, acc.z),
            gyro_sign * Eigen::Vector3f(gyro.x, gyro.y, gyro.z));
        }
        last_main_ukf_imu_stamp_ns_ = rclcpp::Time(imu->header.stamp).nanoseconds();
        ++imu_integrated_count_;
      }

      const double prediction_age = last_main_ukf_imu_stamp_ns_ == 0 ?
        std::numeric_limits<double>::infinity() :
        static_cast<double>(scan_stamp_ns - last_main_ukf_imu_stamp_ns_) * 1e-9;
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "IMU integration: interval=%zu integrated_total=%llu rejected_total=%llu future_queue=%zu prediction_age=%.4f s",
        interval_imu.size(), static_cast<unsigned long long>(imu_integrated_count_),
        static_cast<unsigned long long>(imu_main_ukf_rejected_count_),
        future_queue_size, prediction_age);
    }
    auto t_predict = std::chrono::high_resolution_clock::now();

    if (planar_ndt_enabled_ && !recovery_verification_active_) {
      const bool zero_velocity = stationary_constraint_enabled_ &&
        robot_is_stationary_for_recovery(rclcpp::Time(stamp));
      pose_estimator->apply_planar_constraint(
        stationary_reference_z_, stationary_reference_roll_,
        stationary_reference_pitch_, zero_velocity);
    }

    // odometry-based prediction
    if (!recovery_verification_active_) {
      pose_estimator->reset_odom_prediction();
    }
    if (enable_robot_odometry_prediction && enable_internal_odom_ukf_ &&
        !recovery_verification_active_) {
      if (last_odom_prediction_stamp_ns_ == 0) {
        pose_estimator->predict_odom(Eigen::Matrix4f::Identity());
        last_odom_prediction_stamp_ns_ = scan_stamp_ns;
      } else {
        const rclcpp::Time odom_reference_time(
          last_odom_prediction_stamp_ns_, get_clock()->get_clock_type());
        Eigen::Matrix4f odom_reference_pose;
        Eigen::Matrix4f odom_current_pose;
        if (lookup_odom_pose(odom_reference_time, odom_reference_pose) &&
            lookup_odom_pose(rclcpp::Time(stamp), odom_current_pose)) {
          const Eigen::Matrix4f delta = odom_reference_pose.inverse() * odom_current_pose;
          const float translation = delta.block<2, 1>(0, 3).norm();
          const float yaw_deg = angle_distance(pose_yaw(delta), 0.0f) * 180.0f /
            static_cast<float>(M_PI);
          const double scan_dt = std::max(
            0.0, static_cast<double>(scan_stamp_ns - last_odom_prediction_stamp_ns_) * 1e-9);
          const float max_translation_step = fused_prediction_translation_margin_ +
            fused_prediction_max_speed_ * static_cast<float>(std::min(scan_dt, 0.5));
          if (translation <= max_translation_step &&
              yaw_deg <= fused_prediction_max_yaw_step_deg_) {
            pose_estimator->predict_odom(delta);
          } else {
            RCLCPP_WARN(get_logger(),
              "Discarding implausible fused prediction step: dt=%.3f s xy=%.3f/%.3f m yaw=%.1f/%.1f deg",
              scan_dt, translation, max_translation_step,
              yaw_deg, fused_prediction_max_yaw_step_deg_);
          }
          last_odom_prediction_stamp_ns_ = scan_stamp_ns;
        } else {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 1000,
            "Fused IMU/motor-twist prediction is unavailable for scan stamp; prediction skipped");
        }
      }
    }

    Eigen::Matrix4f deterministic_initial_guess = Eigen::Matrix4f::Identity();
    const bool has_deterministic_initial_guess =
      enable_robot_odometry_prediction &&
      deterministic_fused_initial_guess(
        rclcpp::Time(stamp), deterministic_initial_guess);
    if (enable_robot_odometry_prediction && !has_deterministic_initial_guess &&
        has_reliable_pose_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "Deterministic fused initial guess unavailable; falling back to PoseEstimator prediction");
    }
    // Validate the proposed NDT pose before PoseEstimator is allowed to mutate
    // either UKF. This keeps rejected map/recovery candidates out of the state.
    const bool awaiting_initialization = !is_init_success_;
    const int required_matches = recovery_verification_active_ ?
      recovery_verification_count_threshold_ : init_match_count_threshold_;
    const float acceptance_score_threshold = recovery_verification_active_ ?
      recovery_verification_max_fitness_score_ :
      (awaiting_initialization ? init_match_score_threshold_ : reliable_threshold_);
    bool has_candidate_pose = false;
    std::string candidate_rejection_reason;
    bool has_candidate_recovery_offset = false;
    Eigen::Matrix4f candidate_recovery_offset = Eigen::Matrix4f::Identity();
    float recovery_consistency_max_xy = recovery_max_odom_xy_error_;
    float recovery_consistency_max_yaw_deg = recovery_max_odom_yaw_error_deg_;

    const auto correction_validator =
      [&](const Eigen::Matrix4f& candidate_pose, float score) -> bool {
        has_candidate_pose = true;

        if (!pose_is_map_safe(candidate_pose)) {
          candidate_rejection_reason = "occupancy_pose_gate";
          RCLCPP_WARN(get_logger(),
            "NDT correction rejected before UKF update by occupancy-map pose gate");
          return false;
        }
        if (has_reliable_pose_ &&
            !odom_trajectory_is_map_safe(candidate_pose, scan_stamp_ns)) {
          candidate_rejection_reason = "occupancy_transition_gate";
          RCLCPP_WARN(get_logger(),
            "NDT correction rejected before UKF update: transition crosses occupied or unknown map space");
          return false;
        }
        if (degraded_odom_active_ &&
            !recovery_pose_agrees_with_prediction(
              candidate_pose, rclcpp::Time(stamp),
              &recovery_consistency_max_xy, &recovery_consistency_max_yaw_deg)) {
          const float consistency_xy =
            (candidate_pose.block<2, 1>(0, 3) - degraded_pose_.block<2, 1>(0, 3)).norm();
          const float consistency_yaw_deg = angle_distance(
            pose_yaw(candidate_pose), pose_yaw(degraded_pose_)) *
            180.0f / static_cast<float>(M_PI);
          RCLCPP_WARN(get_logger(),
            "NDT correction rejected before UKF update by fused-prediction consistency gate: score=%.4f xy=%.3f/%.3f m yaw=%.1f/%.1f deg",
            score, consistency_xy, recovery_consistency_max_xy,
            consistency_yaw_deg, recovery_consistency_max_yaw_deg);
          candidate_rejection_reason = "recovery_prediction_gate";
          return false;
        }

        if (recovery_verification_active_) {
          const Eigen::Matrix4f& verification_reference =
            has_deterministic_initial_guess ? deterministic_initial_guess : degraded_pose_;
          candidate_recovery_offset = verification_reference.inverse() * candidate_pose;
          has_candidate_recovery_offset = true;
          if (has_recovery_verification_offset_ &&
              !poses_agree(
                candidate_recovery_offset, recovery_verification_offset_,
                recovery_verification_max_xy_spread_,
                recovery_verification_max_yaw_spread_deg_)) {
            const float spread_xy =
              (candidate_recovery_offset.block<2, 1>(0, 3) -
               recovery_verification_offset_.block<2, 1>(0, 3)).norm();
            const float spread_yaw_deg = angle_distance(
              pose_yaw(candidate_recovery_offset),
              pose_yaw(recovery_verification_offset_)) *
              180.0f / static_cast<float>(M_PI);
            RCLCPP_WARN(get_logger(),
              "NDT correction rejected before UKF update: recovery sequence spread %.3f/%.3f m, %.1f/%.1f deg",
              spread_xy, recovery_verification_max_xy_spread_,
              spread_yaw_deg, recovery_verification_max_yaw_spread_deg_);
            candidate_rejection_reason = "recovery_sequence_gate";
            return false;
          }
        }
        return true;
      };

    const bool stateless_recovery_verification = recovery_verification_active_;
    auto aligned = pose_estimator->correct(
      stamp, raw_points_ptr_, correction_validator,
      has_deterministic_initial_guess ? &deterministic_initial_guess : nullptr,
      !stateless_recovery_verification);
    auto t_correct = std::chrono::high_resolution_clock::now();

    PoseEstimator::MatchResult match_result = pose_estimator->GetMatchState();
      bool match_accepted =
        match_result.is_converged_ &&
        match_result.fitness_score_ < acceptance_score_threshold;
      const auto correction_diagnostics = pose_estimator->correction_diagnostics();
      std::string rejection_reason = correction_diagnostics.rejection_reason;
      if (rejection_reason == "external_gate" && !candidate_rejection_reason.empty()) {
        rejection_reason = candidate_rejection_reason;
      } else if (!match_accepted && rejection_reason.empty()) {
        rejection_reason = "acceptance_score_gate";
      }
      const auto elapsed_ms = [](auto begin, auto end) {
        return static_cast<float>(
          std::chrono::duration<double, std::milli>(end - begin).count());
      };
      publish_scan_matching_status(
        points_msg->header, aligned, correction_diagnostics, match_accepted,
        rejection_reason, input_point_count, filtered_point_count,
        deskew_applied, deskewed_point_count, scan_duration_sec,
        elapsed_ms(start, t_downsample), elapsed_ms(t_downsample, t_predict),
        elapsed_ms(t_predict, t_correct), elapsed_ms(start, t_correct));

      if (awaiting_initialization) {
        localization_state_ = 1;
        if (match_accepted && has_candidate_pose) {
          init_match_count_++;
          if (recovery_verification_active_ && has_candidate_recovery_offset) {
            recovery_verification_offset_ = candidate_recovery_offset;
            has_recovery_verification_offset_ = true;
          }
          RCLCPP_INFO(get_logger(), "Init match count: %d/%d (score: %.6f)",
                      init_match_count_, required_matches, match_result.fitness_score_);
          if (init_match_count_ >= required_matches) {
            if (recovery_verification_active_) {
              const Eigen::Matrix4f verified_pose = correction_diagnostics.candidate_pose;
              last_init_pos_ = verified_pose.block<3, 1>(0, 3);
              last_init_quat_ = Eigen::Quaternionf(verified_pose.block<3, 3>(0, 0));
              pose_estimator.reset(new localization::PoseEstimator(
                registration, rclcpp::Time(stamp), last_init_pos_, last_init_quat_,
                cool_time_duration));
              mark_pose_estimator_recreated();
              last_main_ukf_imu_stamp_ns_ = scan_stamp_ns;
              last_odom_prediction_stamp_ns_ = scan_stamp_ns;
            }
            is_init_success_ = true;
            localization_state_ = 2;
            recovery_count_multiplier_ = 1;
            pose_estimator->set_correction_limits(
              tracking_max_xy_jump_,
              tracking_max_yaw_jump_deg_ * static_cast<float>(M_PI) / 180.0f,
              reliable_threshold_);
            const Eigen::Vector3f cur_pos = pose_estimator->matrix().block<3, 1>(0, 3);
            const float angle = pose_yaw(pose_estimator->matrix()) /
              static_cast<float>(M_PI) * 180.0f;
            RCLCPP_INFO(get_logger(),
              "Init Pose Successful: score=%.6f pos=[%.3f, %.3f, %.3f] angle=[%.3f]",
              match_result.fitness_score_, cur_pos.x(), cur_pos.y(), cur_pos.z(), angle);
            init_match_count_ = 0;
            recovery_verification_active_ = false;
            has_recovery_verification_anchor_ = false;
            has_recovery_verification_offset_ = false;
          }
        } else {
          const bool aborted_recovery_candidate = recovery_verification_active_;
          init_match_count_ = 0;
          recovery_verification_active_ = false;
          has_recovery_verification_anchor_ = false;
          has_recovery_verification_offset_ = false;
          if (aborted_recovery_candidate) {
            is_init_success_ = true;
            RCLCPP_WARN(get_logger(),
              "Stateless recovery verification failed; candidate discarded without changing UKF or map->odom");
          }
          RCLCPP_INFO(get_logger(), "Init match criteria not met, resetting counter");
        }
        RCLCPP_INFO(get_logger(), "Wait Init Pose: current count %d/%d",
                    init_match_count_, required_matches);
      }

      if (enable_robot_odometry_prediction &&
          match_accepted &&
          (!stateless_recovery_verification || is_init_success_)) {
        // A correction synchronizes the odometry UKF with this scan even when
        // the preceding odometry delta was unavailable.
        last_odom_prediction_stamp_ns_ = scan_stamp_ns;
      }
      if (!match_accepted) {
        bad_match_count_++;
        RCLCPP_WARN(get_logger(), "Bad match count: %d/%d (score: %.3f)",
                    bad_match_count_, bad__match_count_threshold_, match_result.fitness_score_);
        const bool transient_tracking_rejection =
          is_init_success_ && !degraded_odom_active_ && has_reliable_pose_ &&
          bad_match_count_ < tracking_degraded_count_threshold_;
        if (transient_tracking_rejection) {
          localization_state_ = 3;
          current_confidence_ = std::min(current_confidence_, 0.5);
          RCLCPP_WARN(get_logger(),
            "Transient NDT rejection %d/%d: retaining fused prediction before DEGRADED_ODOM",
            bad_match_count_, tracking_degraded_count_threshold_);
        } else if (has_reliable_pose_ || degraded_odom_active_) {
          enter_degraded_odom(rclcpp::Time(stamp));
          update_degraded_odom(rclcpp::Time(stamp));
          localization_state_ = 4;
          if (bad_match_count_ == bad__match_count_threshold_) {
            RCLCPP_ERROR(get_logger(),
              "Localization tracking lost; keeping DEGRADED pose continuity while autonomy is blocked");
          }
          start_recovery_search(
            raw_points_ptr_, degraded_pose_, rclcpp::Time(stamp));
        } else {
          // No trusted map pose exists yet. This is still initialization, not
          // a tracking loss, and recovery must not be seeded from identity.
          localization_state_ = 1;
        }
      } else {
        bad_match_count_ = 0;
        if (degraded_odom_active_ && is_init_success_) {
          degraded_odom_active_ = false;
          degraded_odom_blocked_ = false;
          degraded_odom_stationary_episode_ = false;
          RCLCPP_INFO(get_logger(),
            "Leaving DEGRADED_ODOM: recovery passed consecutive-scan verification");
        }
      }

      if (is_init_success_ && match_accepted)
        {
        localization_state_ = 3;
        Eigen::Vector3f cur_pos = pose_estimator->matrix().block<3, 1>(0, 3);
        Eigen::Matrix3f cur_rot = pose_estimator->matrix().block<3, 3>(0, 0);
        float angle = std::atan2(cur_rot(1, 0), cur_rot(0, 0))/M_PI * 180.0f; // Convert to degrees
        RCLCPP_INFO(get_logger(), "Continuous Localization Successful!!! %f pos=[%.3f, %.3f, %.3f] angle=[%.3f]",
                    match_result.fitness_score_, cur_pos.x(), cur_pos.y(), cur_pos.z(), angle);
        }
      else if (degraded_odom_active_)
        {
          localization_state_ = 4;
          RCLCPP_INFO(get_logger(), "Continuous Localization may not good!!! %f", match_result.fitness_score_);
        }
      else if (!is_init_success_)
        {
          localization_state_ = 1;
        }
    auto end = std::chrono::high_resolution_clock::now();
    last_timeout_ = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (last_timeout_ > 80) {
      auto ms = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
      };
      RCLCPP_INFO(get_logger(),
        "!!!point cloud callback time cost > 80ms, = %d ms  (downsample+transform=%ld predict=%ld correct=%ld rest=%ld)\n",
        last_timeout_, ms(start, t_downsample), ms(t_downsample, t_predict), ms(t_predict, t_correct), ms(t_correct, end));
    }

    if (degraded_odom_active_ && has_reliable_pose_) {
      pcl::transformPointCloud(*raw_points_ptr_, *aligned, degraded_pose_);
    }
    if (aligned_pub->get_subscription_count()) {
      aligned->header.frame_id = "map";
      aligned->header.stamp = cloud->header.stamp;
      sensor_msgs::msg::PointCloud2 aligned_msg;
      pcl::toROSMsg(*aligned, aligned_msg);
      aligned_pub->publish(aligned_msg);
    }
    // Update pose history for extrapolation
    if (is_init_success_ && match_accepted) {
      Eigen::Matrix4f current_pose = pose_estimator->matrix();
      Eigen::Vector3f current_velocity = getCurrentVelocity(current_pose, points_msg->header.stamp);
      Eigen::Vector3f current_angular_velocity = getCurrentAngularVelocity(current_pose, points_msg->header.stamp);
      updatePoseHistory(current_pose, current_velocity, current_angular_velocity, points_msg->header.stamp);
      has_initialization_hold_pose_ = false;
      is_extrapolating_ = false;
      current_confidence_ = 1.0;
      last_confidence_update_time_ = points_msg->header.stamp;
      Eigen::Matrix4f reliable_odom_pose;
      if (lookup_odom_pose(rclcpp::Time(points_msg->header.stamp), reliable_odom_pose)) {
        last_reliable_odom_pose_ = reliable_odom_pose;
        has_reliable_odom_pose_ = true;
        last_reliable_scan_stamp_ns_ = scan_stamp_ns;
      } else {
        // Never combine a newly accepted map pose with an older fused-odom
        // anchor. The deterministic predictor resumes after the next accepted
        // scan for which the matching fused pose is available.
        has_reliable_odom_pose_ = false;
        last_reliable_scan_stamp_ns_ = 0;
      }
    }
    const LocalizationOutput output = current_output_locked();
    if (output.available) {
      publish_odometry(
        points_msg->header.stamp, output.pose, true,
        is_init_success_ && match_accepted);
    }
  }

  void reset_tracking_state_locked(
    const rclcpp::Time& stamp,
    bool create_estimator,
    const Eigen::Vector3f& initial_position = Eigen::Vector3f::Zero(),
    const Eigen::Quaternionf& initial_orientation = Eigen::Quaternionf::Identity()) {
    pose_estimator.reset();
    if (create_estimator) {
      pose_estimator.reset(new localization::PoseEstimator(
        registration, stamp, initial_position, initial_orientation,
        cool_time_duration));
      set_initial_correction_limits();
      mark_pose_estimator_recreated();
    }

    is_init_success_ = false;
    localization_state_ = create_estimator ? 1 : 0;
    init_match_count_ = 0;
    bad_match_count_ = 0;
    recovery_count_multiplier_ = 1;
    degraded_odom_active_ = false;
    degraded_odom_blocked_ = false;
    degraded_odom_stationary_episode_ = false;
    recovery_verification_active_ = false;
    has_recovery_verification_anchor_ = false;
    has_recovery_verification_offset_ = false;
    has_reliable_odom_pose_ = false;
    last_reliable_scan_stamp_ns_ = 0;
    has_reliable_pose_ = false;
    has_valid_pose_history_ = false;
    has_initialization_hold_pose_ = create_estimator;
    initialization_hold_pose_ = Eigen::Matrix4f::Identity();
    if (create_estimator) {
      initialization_hold_pose_.block<3, 3>(0, 0) =
        initial_orientation.toRotationMatrix();
      initialization_hold_pose_.block<3, 1>(0, 3) = initial_position;
    }
    last_pose_ = initialization_hold_pose_;
    last_reliable_pose_ = Eigen::Matrix4f::Identity();
    last_reliable_odom_pose_ = Eigen::Matrix4f::Identity();
    degraded_anchor_map_pose_ = Eigen::Matrix4f::Identity();
    degraded_anchor_odom_pose_ = Eigen::Matrix4f::Identity();
    degraded_pose_ = Eigen::Matrix4f::Identity();
    last_velocity_.setZero();
    last_angular_velocity_.setZero();
    last_valid_pose_time_ = stamp;
    current_confidence_ = 0.0;
    last_confidence_update_time_ = stamp;
    is_extrapolating_ = false;
    last_processed_scan_stamp_ns_ = 0;
    last_odom_prediction_stamp_ns_ = 0;
    last_main_ukf_imu_stamp_ns_ = 0;
    last_recovery_search_time_ = rclcpp::Time(0, 0, stamp.get_clock_type());
    global_localization_in_progress_ = false;

    Eigen::Matrix4f bootstrap_map_to_odom = Eigen::Matrix4f::Identity();
    if (create_estimator) {
      Eigen::Matrix4f initial_map_pose = Eigen::Matrix4f::Identity();
      initial_map_pose.block<3, 3>(0, 0) = initial_orientation.toRotationMatrix();
      initial_map_pose.block<3, 1>(0, 3) = initial_position;

      Eigen::Matrix4f current_odom_pose;
      if (lookup_odom_pose(stamp, current_odom_pose)) {
        bootstrap_map_to_odom = initial_map_pose * current_odom_pose.inverse();
      } else {
        // Before the first fused-odometry sample, odom -> base_link is identity.
        bootstrap_map_to_odom = initial_map_pose;
      }
    }

    {
      std::lock_guard<std::mutex> map_to_odom_lock(map_to_odom_mutex_);
      latest_map_to_odom_ = bootstrap_map_to_odom;
      // This provisional transform only makes the TF tree and costmaps
      // observable before initial localization. Health remains non-3, so the
      // control gate still rejects every autonomous velocity command.
      has_map_to_odom_ = publish_bootstrap_map_to_odom_;
    }

    lidar_status_buffer_.assign(static_cast<size_t>(buffer_size_), false);
    {
      std::lock_guard<std::mutex> imu_lock(imu_data_mutex);
      imu_status_buffer_.assign(static_cast<size_t>(buffer_size_), false);
      imu_data.clear();
    }
  }

  /**
   * @brief callback for initial pose input 
   * @param pose_msg
   */
  void initialpose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr pose_msg) {
    RCLCPP_INFO(get_logger(), "initial pose received!!");
    std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    if (recovery_future_.valid()) {
      recovery_future_.wait();
      recovery_future_.get();
    }
  
    const auto& p = pose_msg->pose.pose.position;
    const auto& q = pose_msg->pose.pose.orientation;
    Eigen::Vector3f new_pos(p.x, p.y, p.z);
    Eigen::Quaternionf new_quat(q.w, q.x, q.y, q.z);
    if (new_quat.norm() < 1e-6f) {
      RCLCPP_ERROR(get_logger(), "Rejected initial pose with invalid zero quaternion");
      return;
    }
    new_quat.normalize();
    
    bool pose_changed = false;
    if (!has_set_init_pose_) {
      pose_changed = true;
    } else {
      float pos_change = (new_pos - last_init_pos_).norm();
      float quat_change = std::abs(1.0f - std::abs(last_init_quat_.dot(new_quat))); 
      if (pos_change > init_pose_change_threshold_ || quat_change > init_quat_change_threshold_) {
        pose_changed = true;
        RCLCPP_INFO(get_logger(), "Pose change detected - Position: %.3f m, Orientation: %.3f", pos_change, quat_change);
      }
    }
    
    if (pose_changed) {
      last_init_pos_ = new_pos;
      last_init_quat_ = new_quat;
      has_set_init_pose_ = true;
      last_pose_source_ = "Callback";
      reset_tracking_state_locked(
        get_clock()->now(), true, last_init_pos_, last_init_quat_);
      // A manually supplied pose is the bounded search seed for this round.
      gl_once_gate_ = false;
      RCLCPP_INFO(get_logger(), "New initial pose set from RViz - Position: [%.3f, %.3f, %.3f], Quaternion: [%.3f, %.3f, %.3f, %.3f]",
                   last_init_pos_.x(), last_init_pos_.y(), last_init_pos_.z(), last_init_quat_.w(), last_init_quat_.x(), last_init_quat_.y(), last_init_quat_.z());
      RCLCPP_INFO(get_logger(), "Localization will restart with new pose");
      RCLCPP_INFO(get_logger(), "Initial pose remains untrusted until %d consecutive NDT matches",
                  init_match_count_threshold_);
    } else {
      RCLCPP_INFO(get_logger(), "Pose unchanged, no action needed");
    }
  }

  pcl::PointCloud<PointT>::Ptr downsample(const pcl::PointCloud<PointT>::Ptr& cloud) const {
    if (!voxel_filter_ptr_) {
      return cloud;
    }
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    voxel_filter_ptr_->setInputCloud(cloud);
    voxel_filter_ptr_->filter(*filtered);
    filtered->header = cloud->header;
    return filtered;
  }

  void PublishLocalOdomTimer() {
    if (!enable_robot_odometry_prediction || !local_odom_pub_) return;

    FusedPredictionState state;
    float velocity_x = 0.0f;
    float velocity_y = 0.0f;
    {
      std::lock_guard<std::mutex> lock(fused_prediction_mutex_);
      if (fused_prediction_state_.stamp_ns == 0) return;
      state = fused_prediction_state_;
      const bool motor_twist_fresh = latest_motor_twist_stamp_ns_ != 0 &&
        std::abs(static_cast<double>(state.stamp_ns - latest_motor_twist_stamp_ns_) * 1e-9) <=
          motor_twist_stale_timeout_sec_;
      if (motor_twist_fresh) {
        velocity_x = latest_motor_vx_;
        velocity_y = latest_motor_vy_;
      }
    }

    float yaw_rate = 0.0f;
    {
      std::lock_guard<std::mutex> lock(imu_data_mutex);
      if (correct_imu_data_ptr_) {
        const float gyro_sign = invert_gyro ? -1.0f : 1.0f;
        yaw_rate = gyro_sign * static_cast<float>(
          correct_imu_data_ptr_->angular_velocity.z);
        if (has_cached_imu_biases_) {
          yaw_rate -= cached_gyro_bias_.z();
        }
      }
    }

    Eigen::Matrix4f odom_pose = Eigen::Matrix4f::Identity();
    odom_pose(0, 0) = std::cos(state.yaw);
    odom_pose(0, 1) = -std::sin(state.yaw);
    odom_pose(1, 0) = std::sin(state.yaw);
    odom_pose(1, 1) = std::cos(state.yaw);
    odom_pose(0, 3) = state.x;
    odom_pose(1, 3) = state.y;

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = rclcpp::Time(state.stamp_ns, get_clock()->get_clock_type());
    odom.header.frame_id = robot_odom_frame_id;
    odom.child_frame_id = odom_child_frame_id;
    odom.pose.pose = tf2::toMsg(Eigen::Isometry3d(odom_pose.cast<double>()));
    odom.pose.covariance[0] = 0.04;
    odom.pose.covariance[7] = 0.04;
    odom.pose.covariance[14] = 1e3;
    odom.pose.covariance[21] = 1e3;
    odom.pose.covariance[28] = 1e3;
    odom.pose.covariance[35] = 0.01;

    // The vendor twist is expressed in its fixed odometry axes. ROS Odometry
    // requires twist in child_frame_id, so rotate it into base_link.
    odom.twist.twist.linear.x =
      std::cos(state.yaw) * velocity_x + std::sin(state.yaw) * velocity_y;
    odom.twist.twist.linear.y =
      -std::sin(state.yaw) * velocity_x + std::cos(state.yaw) * velocity_y;
    odom.twist.twist.angular.z = yaw_rate;
    odom.twist.covariance[0] = 0.04;
    odom.twist.covariance[7] = 0.04;
    odom.twist.covariance[14] = 1e3;
    odom.twist.covariance[21] = 1e3;
    odom.twist.covariance[28] = 1e3;
    odom.twist.covariance[35] = 0.01;
    local_odom_pub_->publish(odom);

    if (send_odom_base_transform_) {
      geometry_msgs::msg::TransformStamped transform =
        tf2::eigenToTransform(Eigen::Isometry3d(odom_pose.cast<double>()));
      transform.header.stamp = odom.header.stamp;
      transform.header.frame_id = robot_odom_frame_id;
      transform.child_frame_id = odom_child_frame_id;
      tf_broadcaster->sendTransform(transform);
    }

    // Keep both dynamic transforms on the predictor timeline. NDT updates the
    // map correction at scan time, while this timer republishes that correction
    // beside odom -> base_link so Nav2 never asks for a newer map -> odom sample.
    if (send_tf_transforms) {
      Eigen::Matrix4f map_to_odom;
      bool available = false;
      {
        std::lock_guard<std::mutex> lock(map_to_odom_mutex_);
        if (has_map_to_odom_) {
          map_to_odom = latest_map_to_odom_;
          available = true;
        }
      }
      if (available) {
        geometry_msgs::msg::TransformStamped transform =
          tf2::eigenToTransform(Eigen::Isometry3d(map_to_odom.cast<double>()));
        transform.header.stamp = odom.header.stamp;
        transform.header.frame_id = "map";
        transform.child_frame_id = robot_odom_frame_id;
        tf_broadcaster->sendTransform(transform);
      }
    }
  }

  void publish_odometry(
    const rclcpp::Time& stamp, const Eigen::Matrix4f& pose,
    bool pose_available = true, bool update_map_to_odom = false) {
    // RCLCPP_INFO(
    //   get_logger(),
    //   "[publish_odometry] stamp_ns=%ld now_ns=%ld send_tf_transforms=%s frame_id=%s child_frame_id=%s pose_xyz=[%.3f, %.3f, %.3f]",
    //   static_cast<long>(stamp.nanoseconds()),
    //   static_cast<long>(get_clock()->now().nanoseconds()),
    //   send_tf_transforms ? "true" : "false",
    //   robot_odom_frame_id.c_str(),
    //   localization_odom_frame_id.c_str(),
    //   pose(0, 3), pose(1, 3), pose(2, 3));
    if (send_tf_transforms && update_map_to_odom) {
      Eigen::Matrix4f odom_pose;
      if (lookup_odom_pose(stamp, odom_pose)) {
        const Eigen::Matrix4f map_to_odom = pose * odom_pose.inverse();
        {
          std::lock_guard<std::mutex> lock(map_to_odom_mutex_);
          latest_map_to_odom_ = map_to_odom;
          has_map_to_odom_ = true;
        }
        if (!enable_robot_odometry_prediction) {
          geometry_msgs::msg::TransformStamped transform =
            tf2::eigenToTransform(Eigen::Isometry3d(map_to_odom.cast<double>()));
          transform.header.stamp = stamp;
          transform.header.frame_id = "map";
          transform.child_frame_id = robot_odom_frame_id;
          tf_broadcaster->sendTransform(transform);
        }
      } else {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "[publish_odometry] fused odometry is unavailable at pose stamp; map -> %s TF not published",
          robot_odom_frame_id.c_str());
      }
    }
    // publish the transform
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "map";
    odom.pose.pose = tf2::toMsg(Eigen::Isometry3d(pose.cast<double>()));
    odom.child_frame_id = localization_odom_frame_id;
    const auto covariance =
      output_pose_covariance_locked(
        current_localization_status_locked(), pose_available);
    std::copy(covariance.begin(), covariance.end(), odom.pose.covariance.begin());
    odom.twist.twist.linear.x = 0.0;
    odom.twist.twist.linear.y = 0.0;
    odom.twist.twist.angular.z = 0.0;
    // Реальная скорость из UKF вместо нулей — чтобы MPPI/progress checker знали,
    // с какой скоростью робот уже едет. UKF хранит скорость в map-фрейме,
    // а twist в Odometry должен быть в body-фрейме (child_frame_id=base_link),
    // поэтому поворачиваем через R^T. Guard на pose_estimator обязателен:
    // publish_odometry вызывается и до его создания (pubDefaultLocalizationOdom).
    // if (pose_estimator) {
    //   const Eigen::Matrix3f R = pose.block<3, 3>(0, 0);
    //   const Eigen::Vector3f vel_body = R.transpose() * pose_estimator->vel();
    //   odom.twist.twist.linear.x = vel_body.x();
    //   odom.twist.twist.linear.y = vel_body.y();
    // }
    pose_pub->publish(odom);
    // RCLCPP_INFO(
    //   get_logger(),
    //   "[publish_odometry] published odom header_ns=%ld frame_id=%s child_frame_id=%s",
    //   static_cast<long>(static_cast<long long>(odom.header.stamp.sec) * 1000000000LL + odom.header.stamp.nanosec),
    //   odom.header.frame_id.c_str(),
    //   odom.child_frame_id.c_str());
  }

  /**
   * @brief Publish default localization information (position is 0)
   * @param stamp timestamp
   */
  void pubDefaultLocalizationOdom(const rclcpp::Time& stamp) {
    is_extrapolating_ = false;
    current_confidence_ = 0.0;
    Eigen::Matrix4f default_pose = Eigen::Matrix4f::Identity();
    publish_odometry(stamp, default_pose, false);
  }

  /**
   * @brief perform global localization
   * @param current_cloud current point cloud
   * @return true if global localization is successful
   */
  bool performGlobalLocalization(const pcl::PointCloud<PointT>::Ptr& current_cloud) {
    if (!global_localization_ptr_ || !global_map_points_ptr_) {
      RCLCPP_WARN(get_logger(), "Global localization not available");
      return false;
    }
    global_localization_in_progress_ = true;
    global_localization_start_time_ = get_clock()->now();  
    RCLCPP_INFO(get_logger(), "Starting global localization..."); 
    try {
      Eigen::Vector3f init_pos;
      Eigen::Quaternionf init_quat;
      if (!getCurrentInitPose(init_pos, init_quat)) {
        RCLCPP_WARN(get_logger(), "Failed to get initial pose for global localization");
        global_localization_in_progress_ = false;
        return false;
      }
      
      Eigen::Matrix4d initial_trans = Eigen::Matrix4d::Identity();
      initial_trans.block<3, 1>(0, 3) = init_pos.cast<double>();
      initial_trans.block<3, 3>(0, 0) = init_quat.toRotationMatrix().cast<double>();
      
      Eigen::Matrix4d final_pose;
      bool localization_success = global_localization_ptr_->performGlobalLocalization(
        global_map_points_ptr_, current_cloud, initial_trans, final_pose);
      
      if (localization_success) {
        Eigen::Vector3f new_pos = final_pose.block<3, 1>(0, 3).cast<float>();
        Eigen::Quaternionf new_quat(final_pose.block<3, 3>(0, 0).cast<float>());
        if (pose_estimator) {
          last_init_pos_ = new_pos;
          last_init_quat_ = new_quat;
          has_set_init_pose_ = true;
          last_pose_source_ = "global_localization";
          RCLCPP_INFO(get_logger(), "Global localization successful! New pose: pos[%.3f, %.3f, %.3f], quat[%.3f, %.3f, %.3f, %.3f]",
                      new_pos.x(), new_pos.y(), new_pos.z(), new_quat.w(), new_quat.x(), new_quat.y(), new_quat.z());
          global_localization_in_progress_ = false;
          return true;
        }
      } else {
        RCLCPP_WARN(get_logger(), "Global localization failed");
      }
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Global localization exception: %s", e.what());
    }
    global_localization_in_progress_ = false;
    return false;
  }
  
  /**
   * @brief get current initial pose
   * @param pos output position
   * @param quat output orientation
   * @return true if success
   */
  bool getCurrentInitPose(Eigen::Vector3f& pos, Eigen::Quaternionf& quat) {
    if (has_set_init_pose_) {
      pos = last_init_pos_;
      quat = last_init_quat_;
      return true;
    }
    if (specify_init_pose_) {
      pos = Eigen::Vector3f(init_pos_x_, init_pos_y_, init_pos_z_);
      quat = Eigen::Quaternionf(init_ori_w_, init_ori_x_, init_ori_y_, init_ori_z_);
      return true;
    }
    pos = Eigen::Vector3f::Zero();
    quat = Eigen::Quaternionf::Identity();
    return true;
  }
  
  /**
   * @brief Check if lidar data is valid
   * @return true: lidar data is valid, false: lidar data is invalid
   */
  bool isLidarDataValid() {
    if (lidar_status_buffer_.size() < min_valid_count_) { return false; }
    int valid_count = std::count(lidar_status_buffer_.begin(), lidar_status_buffer_.end(), true);
    return valid_count >= min_valid_count_;
  }

  /**
   * @brief Check if IMU data is valid
   * @return true: IMU data is valid, false: IMU data is invalid
   */
  bool isImuDataValid() {
    if (!use_imu) { return false; }
    std::lock_guard<std::mutex> lock(imu_data_mutex);  // imu_status_buffer_ written from imu_group thread
    if (imu_status_buffer_.size() < min_valid_count_) { return false; }
    int valid_count = std::count(imu_status_buffer_.begin(), imu_status_buffer_.end(), true);
    return valid_count >= min_valid_count_;
  }

  /**
   * @brief Check if sensor data is valid (at least one of lidar or IMU is valid)
   * @return true: at least one sensor data is valid, false: all sensor data are invalid
   */
  bool isSensorDataValid() { return isLidarDataValid() || isImuDataValid(); }

  /**
   * @brief Update lidar data status
   * @param is_valid whether data is valid
   */
  void updateLidarStatus(bool is_valid) {
    lidar_status_buffer_.push_back(is_valid);
    if (lidar_status_buffer_.size() > buffer_size_) {
      lidar_status_buffer_.pop_front();
    }
    if (is_valid) {
      last_lidar_data_time_ = get_clock()->now();
    }
  }

  /**
   * @brief Update IMU data status
   * @param is_valid whether data is valid
   */
  void updateImuStatus(bool is_valid) {
    imu_status_buffer_.push_back(is_valid);
    if (imu_status_buffer_.size() > buffer_size_) {
      imu_status_buffer_.pop_front();
    }
    if (is_valid) {
      last_imu_data_time_ = get_clock()->now();
    }
  }
  
  // Transform points using gravity transformation matrix 
  void TransformPoints(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in, pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_out) {
      cloud_out->clear();
      int point_size = cloud_in->points.size();
      cloud_out->resize(point_size);
  #pragma omp parallel for
      for (int i = 0; i < cloud_in->points.size(); ++i) {
          pcl::PointXYZI  point;
          Eigen::Vector4f p_start(cloud_in->points[i].x, cloud_in->points[i].y, cloud_in->points[i].z, 1.0);
          Eigen::Vector4f p_result(gravity_transform_ * p_start);
          point.x              = p_result(0);
          point.y              = p_result(1);
          point.z              = p_result(2);
          point.intensity      = cloud_in->points[i].intensity;
          cloud_out->points[i] = point;
      }
  }

  /**
   * @brief Update pose history for extrapolation
   * @param pose current pose matrix
   * @param velocity current velocity
   * @param angular_velocity current angular velocity
   * @param current_time current time
   */
  void updatePoseHistory(const Eigen::Matrix4f& pose, const Eigen::Vector3f& velocity, 
                         const Eigen::Vector3f& angular_velocity, const rclcpp::Time& current_time) {
    last_pose_ = pose;
    last_reliable_pose_ = pose;
    has_reliable_pose_ = true;
    last_velocity_ = velocity;
    last_angular_velocity_ = angular_velocity;
    last_valid_pose_time_ = current_time;
    has_valid_pose_history_ = true;
  }

  /**
   * @brief Get current velocity (prioritize UKF state, extrapolation as backup)
   * @param current_pose current pose
   * @param current_time current time
   * @return current velocity
   */
  Eigen::Vector3f getCurrentVelocity(const Eigen::Matrix4f& current_pose, const rclcpp::Time& current_time) {
    if (pose_estimator) {
      return pose_estimator->vel();
    }
    if (has_valid_pose_history_) {
      double dt = (current_time - last_valid_pose_time_).seconds();
      if (dt > 0.0) {
        Eigen::Vector3f current_position = current_pose.block<3, 1>(0, 3);
        Eigen::Vector3f last_position = last_pose_.block<3, 1>(0, 3);
        Eigen::Vector3f position_delta = current_position - last_position;
        return position_delta / dt;
      }
    }
    return last_velocity_;
  }

  /**
   * @brief Get current angular velocity (prioritize IMU data, extrapolation as backup)
   * @param current_pose current pose
   * @param current_time current time
   * @return current angular velocity
   */
  Eigen::Vector3f getCurrentAngularVelocity(const Eigen::Matrix4f& current_pose, const rclcpp::Time& current_time) {
    {
      std::lock_guard<std::mutex> lock(imu_data_mutex);  // latest_angular_velocity_ written from imu_group thread
      if (latest_angular_velocity_.norm() > 0.0) {
        return latest_angular_velocity_;
      }
    }
    if (has_valid_pose_history_) {
      double dt = (current_time - last_valid_pose_time_).seconds();
      if (dt > 0.0) {
        Eigen::Matrix3f current_rotation = current_pose.block<3, 3>(0, 0);
        Eigen::Matrix3f last_rotation = last_pose_.block<3, 3>(0, 0);
        Eigen::Matrix3f relative_rotation = current_rotation * last_rotation.transpose();
        Eigen::AngleAxisf angle_axis(relative_rotation);
        Eigen::Vector3f angular_velocity = angle_axis.axis() * angle_axis.angle() / dt;
        return angular_velocity;
      }
    }
    return last_angular_velocity_;
  }

  /**
   * @brief Get sensor status statistics information
   * @return string containing sensor status statistics information
   */
  std::string getSensorStatusInfo() const {
    std::stringstream ss;
    // Lidar status statistics
    int lidar_valid_count = std::count(lidar_status_buffer_.begin(), lidar_status_buffer_.end(), true);
    int lidar_total_count = lidar_status_buffer_.size();
    double lidar_valid_ratio = lidar_total_count > 0 ? (double)lidar_valid_count / lidar_total_count : 0.0; 
    // IMU status statistics
    int imu_valid_count, imu_total_count;
    {
      std::lock_guard<std::mutex> lock(imu_data_mutex);  // imu_status_buffer_ written from imu_group thread
      imu_valid_count = std::count(imu_status_buffer_.begin(), imu_status_buffer_.end(), true);
      imu_total_count = imu_status_buffer_.size();
    }
    double imu_valid_ratio = imu_total_count > 0 ? (double)imu_valid_count / imu_total_count : 0.0;
    
    ss << "Lidar: " << lidar_valid_count << "/" << lidar_total_count 
       << " (" << std::fixed << std::setprecision(1) << (lidar_valid_ratio * 100.0) << "%)"
       << " | IMU: " << imu_valid_count << "/" << imu_total_count 
       << " (" << std::fixed << std::setprecision(1) << (imu_valid_ratio * 100.0) << "%)"
       << " | Pose History: " << (has_valid_pose_history_ ? "Available" : "Not Available")
       << " | Confidence: " << std::fixed << std::setprecision(2) << current_confidence_; 
    return ss.str();
  }

  /**
   * @brief Update confidence
   * @param current_time current time
   */
  void updateConfidence(const rclcpp::Time& current_time) {
    if (!is_init_success_) {
      current_confidence_ = 0.0;
    } else if (is_extrapolating_) {
      // When extrapolating, confidence decays exponentially
      double dt = (current_time - last_confidence_update_time_).seconds();
      if (dt > 0.0) {
        current_confidence_ *= std::exp(-confidence_decay_rate_ * dt);
        if (current_confidence_ < 0.01) {
          current_confidence_ = 0.01;
        }
      }
    } else {
      current_confidence_ = 1.0;
    } 
    last_confidence_update_time_ = current_time;
  }

  /**
   * @brief Get current confidence
   * @return current confidence value
   */
  double getCurrentConfidence() const { return current_confidence_; }

  /**
   * @brief Control log printing frequency
   * @param message message to print
   * @param force_print whether to force print (ignore counter)
   */
  void controlledLogInfo(const std::string& message, bool force_print = false) {
    log_counter_++;
    if (force_print || log_counter_ % log_interval_ == 0) {
      RCLCPP_INFO(get_logger(), "%s", message.c_str());
      log_counter_ = 0;  // Reset counter
    }
  }

  /**
   * @brief Convert quaternion to normalized Euler angles (ZYX order, consistent with Eigen)
   * @param quat quaternion (w, x, y, z)
   * @return normalized Euler angles [roll, pitch, yaw] (ZYX order)
   */
  Eigen::Vector3f quaternionToNormalizedRPY(const Eigen::Quaternionf& quat) {
    Eigen::Quaternionf normalized_quat = quat.normalized();
    float w = normalized_quat.w();
    float x = normalized_quat.x();
    float y = normalized_quat.y();
    float z = normalized_quat.z();
    // 0=Z-axis(yaw), 1=Y-axis(pitch), 2=X-axis(roll)
    float roll, pitch, yaw;
    // Calculate Roll 
    float sinr_cosp = 2.0f * (w * x + y * z);
    float cosr_cosp = 1.0f - 2.0f * (x * x + y * y);
    roll = std::atan2(sinr_cosp, cosr_cosp);
    // Calculate Pitch 
    float sinp = 2.0f * (w * y - x * z);
    if (std::abs(sinp) >= 1.0f) {
      // Handle gimbal lock case (pitch = ±90°)
      pitch = std::copysign(M_PI / 2.0f, sinp);
      roll = 0.0f;
      yaw = 2.0f * std::atan2(x, w);
    } else {
      pitch = std::asin(sinp);
      // Calculate Yaw 
      float siny_cosp = 2.0f * (w * z + x * y);
      float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
      yaw = std::atan2(siny_cosp, cosy_cosp);
    }

    if (roll > M_PI / 2.0f) {
      roll -= M_PI;
      pitch = M_PI - pitch;
      yaw += M_PI;
    } else if (roll < -M_PI / 2.0f) {
      roll += M_PI;
      pitch = -M_PI - pitch;
      yaw += M_PI;
    }
    if (pitch > M_PI / 2.0f) {
      pitch = M_PI - pitch;
      roll += M_PI;
      yaw += M_PI;
    } else if (pitch < -M_PI / 2.0f) {
      pitch = -M_PI - pitch;
      roll += M_PI;
      yaw += M_PI;
    }
    if (yaw > M_PI) {
      yaw -= 2.0f * M_PI;
    } else if (yaw < -M_PI) {
      yaw += 2.0f * M_PI;
    }
    return Eigen::Vector3f(roll, pitch, yaw);
  }

  /**
   * @brief Extrapolate pose based on previous state
   * @param current_time current time
   * @return extrapolated pose matrix
   */
  Eigen::Matrix4f extrapolatePose(const rclcpp::Time& current_time) {
    if (!has_valid_pose_history_) {
      return Eigen::Matrix4f::Identity();
    }
    // Use fixed time step 0.05 seconds (20Hz)
    const double dt = 0.05;
    const double max_velocity = 1.0;        // Maximum velocity
    const double max_angular_velocity = 0.5; // Maximum angular velocity
    Eigen::Vector3f limited_velocity = last_velocity_;
    double velocity_norm = limited_velocity.norm();
    if (velocity_norm > max_velocity) {
      limited_velocity = limited_velocity.normalized() * max_velocity;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1.0, 
                           "Velocity limited from %.2f to %.2f m/s", velocity_norm, max_velocity);
    }
    
    // Limit angular velocity range
    Eigen::Vector3f limited_angular_velocity = last_angular_velocity_;
    double angular_velocity_norm = limited_angular_velocity.norm();
    if (angular_velocity_norm > max_angular_velocity) {
      limited_angular_velocity = limited_angular_velocity.normalized() * max_angular_velocity;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1.0, 
                           "Angular velocity limited from %.2f to %.2f rad/s", angular_velocity_norm, max_angular_velocity);
    }
    Eigen::Vector3f position_delta = limited_velocity * dt;
    Eigen::Vector3f last_position = last_pose_.block<3, 1>(0, 3);
    Eigen::Matrix3f last_rotation = last_pose_.block<3, 3>(0, 0);
    Eigen::Vector3f new_position = last_position + position_delta;
    Eigen::Vector3f angle_delta = limited_angular_velocity * dt;
    Eigen::Matrix3f delta_rotation = Eigen::Matrix3f::Identity();
    delta_rotation = Eigen::AngleAxisf(angle_delta.z(), Eigen::Vector3f::UnitZ()) *
                     Eigen::AngleAxisf(angle_delta.y(), Eigen::Vector3f::UnitY()) *
                     Eigen::AngleAxisf(angle_delta.x(), Eigen::Vector3f::UnitX());
    Eigen::Matrix3f new_rotation = last_rotation * delta_rotation;
    Eigen::Matrix4f extrapolated_pose = Eigen::Matrix4f::Identity();
    extrapolated_pose.block<3, 3>(0, 0) = new_rotation;
    extrapolated_pose.block<3, 1>(0, 3) = new_position;
    
    // Debug information
    RCLCPP_DEBUG(get_logger(), 
                 "Extrapolation: dt=%.3fs, vel=[%.2f,%.2f,%.2f], pos_delta=[%.2f,%.2f,%.2f], "
                 "new_pos=[%.2f,%.2f,%.2f]", 
                 dt, 
                 limited_velocity.x(), limited_velocity.y(), limited_velocity.z(),
                 position_delta.x(), position_delta.y(), position_delta.z(),
                 new_position.x(), new_position.y(), new_position.z());
    
    return extrapolated_pose;
  }

  /**
   * @brief Publish extrapolated localization information
   * @param stamp timestamp
   */
  void publishExtrapolatedOdom(const rclcpp::Time& stamp) {
    if (degraded_odom_active_ && has_reliable_pose_) {
      update_degraded_odom(stamp);
      current_confidence_ = 0.0;
      publish_odometry(stamp, degraded_pose_);
      return;
    }
    if (!has_valid_pose_history_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1.0, 
                           "No pose history available, publishing default pose");
      pubDefaultLocalizationOdom(stamp);
      return;
    }
    // Set extrapolation state to true, start confidence decay
    is_extrapolating_ = true;
    Eigen::Matrix4f extrapolated_pose = extrapolatePose(stamp);
    publish_odometry(stamp, extrapolated_pose);
    const double extrapolation_time = (stamp - last_valid_pose_time_).seconds();
    last_pose_ = extrapolated_pose;
    last_valid_pose_time_ = stamp;
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1.0, 
                         "Published extrapolated pose and updated history (extrapolation time: %.2fs, "
                         "last velocity: [%.2f, %.2f, %.2f], "
                         "last angular velocity: [%.2f, %.2f, %.2f], "
                         "confidence: %.3f)", 
                         extrapolation_time,
                         last_velocity_.x(), last_velocity_.y(), last_velocity_.z(),
                         last_angular_velocity_.x(), last_angular_velocity_.y(), last_angular_velocity_.z(),
                         getCurrentConfidence());
  }

  void publish_scan_matching_status(
      const std_msgs::msg::Header& header,
      pcl::PointCloud<pcl::PointXYZI>::ConstPtr aligned,
      const PoseEstimator::CorrectionDiagnostics& diagnostics,
      bool accepted,
      const std::string& rejection_reason,
      size_t input_points,
      size_t filtered_points,
      bool deskew_applied,
      uint32_t deskewed_points,
      double scan_duration_sec,
      float downsample_ms,
      float prediction_ms,
      float registration_ms,
      float total_callback_ms) {
    localization::msg::ScanMatchingStatus status;
    status.header = header;
    status.has_converged = registration->hasConverged();
    status.accepted = accepted;
    status.rejection_reason = accepted ? "" : rejection_reason;
    status.matching_error = static_cast<float>(registration->getFitnessScore());
    const double max_correspondence_dist = 0.5;

    size_t num_inliers = 0;
    size_t checked_points = 0;
    std::vector<int> k_indices;
    std::vector<float> k_sq_dists;
    constexpr size_t max_overlap_samples = 512;
    if (aligned && !aligned->empty() && registration->getSearchMethodTarget()) {
      const size_t stride = std::max<size_t>(1, aligned->size() / max_overlap_samples);
      for (size_t i = 0; i < aligned->size(); i += stride) {
        const auto& pt = aligned->at(i);
        if (registration->getSearchMethodTarget()->nearestKSearch(
              pt, 1, k_indices, k_sq_dists) > 0 &&
            !k_sq_dists.empty() &&
            k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist) {
          ++num_inliers;
        }
        ++checked_points;
      }
    }
    status.inlier_fraction = checked_points == 0 ? 0.0f :
      static_cast<float>(num_inliers) / static_cast<float>(checked_points);
    status.initial_guess = tf2::eigenToTransform(
      Eigen::Isometry3d(diagnostics.initial_guess.cast<double>())).transform;
    status.candidate_pose = tf2::eigenToTransform(
      Eigen::Isometry3d(diagnostics.candidate_pose.cast<double>())).transform;
    status.innovation = tf2::eigenToTransform(
      Eigen::Isometry3d(diagnostics.innovation.cast<double>())).transform;
    status.relative_pose = status.innovation;

    const int64_t scan_stamp_ns = rclcpp::Time(header.stamp).nanoseconds();
    status.scan_age_sec = std::max(
      0.0, static_cast<double>(get_clock()->now().nanoseconds() - scan_stamp_ns) * 1e-9);
    {
      std::lock_guard<std::mutex> lock(imu_data_mutex);
      status.imu_prediction_age_sec = last_main_ukf_imu_stamp_ns_ == 0 ? -1.0 :
        static_cast<double>(scan_stamp_ns - last_main_ukf_imu_stamp_ns_) * 1e-9;
      status.imu_received = imu_received_count_;
      status.imu_integrated = imu_integrated_count_;
      status.imu_duplicates = imu_duplicate_count_;
      status.imu_out_of_order = imu_out_of_order_count_;
      status.imu_buffer_overflow = imu_buffer_overflow_count_;
    }
    {
      std::lock_guard<std::mutex> lock(fused_prediction_mutex_);
      status.odom_prediction_age_sec = latest_motor_twist_stamp_ns_ == 0 ? -1.0 :
        static_cast<double>(scan_stamp_ns - latest_motor_twist_stamp_ns_) * 1e-9;
      status.motor_odom_received = motor_odom_received_count_;
      status.motor_odom_duplicates = motor_odom_duplicate_count_;
      status.motor_odom_out_of_order = motor_odom_out_of_order_count_;
    }
    status.input_points = static_cast<uint32_t>(std::min<size_t>(
      input_points, std::numeric_limits<uint32_t>::max()));
    status.filtered_points = static_cast<uint32_t>(std::min<size_t>(
      filtered_points, std::numeric_limits<uint32_t>::max()));
    status.deskew_applied = deskew_applied;
    status.deskewed_points = deskewed_points;
    status.scan_duration_sec = scan_duration_sec;
    status.downsample_ms = downsample_ms;
    status.prediction_ms = prediction_ms;
    status.registration_ms = registration_ms;
    status.total_callback_ms = total_callback_ms;
    status.prediction_labels.reserve(2);
    status.prediction_errors.reserve(2);
    if (pose_estimator->wo_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::msg::String());
      status.prediction_labels.back().data = "without_pred";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->wo_prediction_error().get().cast<double>())).transform);
    }
    if (pose_estimator->imu_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::msg::String());
      status.prediction_labels.back().data = use_imu ? "imu" : "motion_model";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->imu_prediction_error().get().cast<double>())).transform);
    }
    if (pose_estimator->odom_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::msg::String());
      status.prediction_labels.back().data = "odom";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->odom_prediction_error().get().cast<double>())).transform);
    }
    status_pub->publish(status);
  }

  void publish_localization_interfaces(
      const robots_dog_msgs::msg::Localization& health_msg) {
        localization_health_pub_->publish(health_msg);
        if (publish_vendor_localization_info_) {
            // Legacy output is optional and carries the real state. MO TF and
            // command safety never depend on the vendor robot_tf authority.
            localization_info_pub_->publish(health_msg);
        }
  }

  void PublishLidarLocalizationInfo() {
        auto current_time = this->get_clock()->now();
        if (lidar_status_buffer_.size() > 0) {
            double lidar_time_diff = (current_time - last_lidar_data_time_).seconds();
            if (lidar_time_diff > sensor_timeout_threshold_) {
                updateLidarStatus(false);
            }
        }  
        {
            std::lock_guard<std::mutex> lock(imu_data_mutex);  // imu_status_buffer_/last_imu_data_time_ written from imu_group thread
            if (imu_status_buffer_.size() > 0) {
                double imu_time_diff = (current_time - last_imu_data_time_).seconds();
                if (imu_time_diff > sensor_timeout_threshold_) {
                    updateImuStatus(false);
                }
            }
        }
        robots_dog_msgs::msg::Localization msg;
        msg.header.stamp = this->get_clock()->now();
        msg.header.frame_id = "map";
        msg.type = "loc_state";
        const bool sensors_valid = isSensorDataValid();
        LocalizationOutput output;
        {
            std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex);
            updateConfidence(current_time);
            output = current_output_locked();
            if (!sensors_valid) {
                output.status = 4;
                if (has_valid_pose_history_) {
                    output.pose = last_pose_;
                    output.velocity = last_velocity_;
                    output.available = true;
                }
            }
        }
        msg.status = output.status;
        msg.coord_type = is_use_map_coord_ ? 0 : 1; // 0 map coordinate, 1 latitude and longitude coordinate
        if (output.available) {
            const Eigen::Vector3f position = output.pose.block<3, 1>(0, 3);
            if (msg.coord_type == 0) {
                msg.pos.x = position.x();
                msg.pos.y = position.y();
                msg.pos.z = position.z();
            }
            const Eigen::Quaternionf quat(output.pose.block<3, 3>(0, 0));
            Eigen::Vector3f rpy = quaternionToNormalizedRPY(quat);
            msg.rpy.x = rpy(0); // roll
            msg.rpy.y = rpy(1); // pitch
            msg.rpy.z = rpy(2); // yaw
        } else {
            msg.pos.x = msg.pos.y = msg.pos.z = 0.0;
            msg.rpy.x = msg.rpy.y = msg.rpy.z = 0.0;
        }
        sensor_msgs::msg::Imu::SharedPtr imu_snapshot;
        {
            std::lock_guard<std::mutex> lock(imu_data_mutex);  // correct_imu_data_ptr_ written from imu_group thread
            imu_snapshot = correct_imu_data_ptr_;
        }
        if (imu_snapshot) {
            const auto &imu = *imu_snapshot;
            msg.acc.x = imu.linear_acceleration.x;
            msg.acc.y = imu.linear_acceleration.y;
            msg.acc.z = imu.linear_acceleration.z;

            msg.gyro.x = imu.angular_velocity.x;
            msg.gyro.y = imu.angular_velocity.y;
            msg.gyro.z = imu.angular_velocity.z;
        } else {
            msg.acc.x = msg.acc.y = msg.acc.z = 0.0;
            msg.gyro.x = msg.gyro.y = msg.gyro.z = 0.0;
        }
        msg.vel.x = output.velocity(0);
        msg.vel.y = output.velocity(1);
        msg.vel.z = output.velocity(2);
        msg.speed = output.velocity.norm();
        publish_localization_interfaces(msg);
        controlledLogInfo("Sensor Status: " + getSensorStatusInfo());
    }

    /**
     * @brief Odometry publishing timer callback function
     * Only provides supplementary odometry publishing when sensors fail or not initialized, does not interfere with original logic
     */
  void PublishOdomTimer() {
        std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex);
        if (!is_init_success_) {
            if (degraded_odom_active_ && has_reliable_pose_) {
                update_degraded_odom(this->get_clock()->now());
                current_confidence_ = 0.0;
                publish_odometry(this->get_clock()->now(), degraded_pose_);
            } else if (has_initialization_hold_pose_) {
                current_confidence_ = 0.0;
                publish_odometry(this->get_clock()->now(), initialization_hold_pose_);
            } else if (has_reliable_pose_) {
                current_confidence_ = 0.0;
                publish_odometry(this->get_clock()->now(), last_reliable_pose_);
            } else {
                RCLCPP_DEBUG(get_logger(), "Localization not initialized and no pose is available");
            }
            return;
        }  

        if (!isSensorDataValid()) {
            RCLCPP_DEBUG(get_logger(), "Sensor data invalid, checking pose history...");
            auto current_time = this->get_clock()->now();
            if (has_valid_pose_history_) {
                double time_since_last_pose = (current_time - last_valid_pose_time_).seconds();
                RCLCPP_DEBUG(get_logger(), "Time since last pose: %.2fs, using extrapolation", time_since_last_pose);
                
                publishExtrapolatedOdom(current_time);
                return;
            } else {
                RCLCPP_DEBUG(get_logger(), "No valid pose history available, using default pose");
                pubDefaultLocalizationOdom(current_time);
                return;
            }
        }
        if (is_extrapolating_) {
            is_extrapolating_ = false;
            current_confidence_ = 1.0;
            RCLCPP_DEBUG(get_logger(), "Sensor data recovered, resetting extrapolation state, confidence: 1.0");
        }
        RCLCPP_DEBUG(get_logger(), "Sensor data valid, no supplementary odometry needed");
    }

    void LocalizationStateCallback(const std::shared_ptr<robots_dog_msgs::srv::LocalizationState::Request> request,
            std::shared_ptr<robots_dog_msgs::srv::LocalizationState::Response> response) {
         uint8_t receive_message = request->data;
        switch (receive_message) {
        case 0: 
            mode_state_.store(ModeState::INIT);
            response->success = true;
            response->message = "Initialized (Mode: INIT)";
            break;
        case 2:
            mode_state_.store(ModeState::READY);
            response->success = true;
            response->message = "Set READY State (will auto switch to ACTIVE when ready)";
            break;
        case 4: 
            mode_state_.store(ModeState::SUCCESS);
            response->success = true;
            response->message = "Localization stopped (Mode: SUCCESS)";
            break;
        default:
            response->success = false;
            response->message = "Invalid State";
            break;
        }
    }

    void LoadMapCallBack(robots_dog_msgs::srv::LoadMap::Request::SharedPtr request, 
            robots_dog_msgs::srv::LoadMap::Response::SharedPtr response) {
        update_map_flag_.store(true);
        std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex);
        if (recovery_future_.valid()) {
          recovery_future_.wait();
          recovery_future_.get();
        }
        std::string map_path = request->pcd_path;

        if (!std::filesystem::exists(std::filesystem::path(map_path))) {
            RCLCPP_ERROR(get_logger(), "Map file does not exist: %s", map_path.c_str());
            response->success = false;
            response->message = "Map file does not exist.";
            return;
        }
        global_map_points_ptr_->clear();
        pcl::io::loadPCDFile(map_path, *global_map_points_ptr_);
        RCLCPP_INFO(get_logger(), "Global map points size==: %zu", global_map_points_ptr_->points.size());
        if (!global_map_points_ptr_->empty()) {
            pcl::VoxelGrid<pcl::PointXYZI> voxel;
            voxel.setInputCloud(global_map_points_ptr_);
            voxel.setLeafSize(globalmap_voxel_size_, globalmap_voxel_size_, globalmap_voxel_size_);
            voxel.filter(*global_map_points_ptr_);
            if (global_map_points_ptr_->points.size() < 1000) {
                RCLCPP_ERROR(get_logger(), "Global map points size is too small: %zu", global_map_points_ptr_->points.size());
                response->success = false;
                response->message = "Global map points size is too small.";
                return;
            } else {
                if (global_map_pub_->get_subscription_count()) {
                    pcl::PointCloud<pcl::PointXYZI>::Ptr global_map_downsampled(new pcl::PointCloud<pcl::PointXYZI>);
                    pcl::VoxelGrid<pcl::PointXYZI> publish_voxel;
                    publish_voxel.setInputCloud(global_map_points_ptr_);
                    publish_voxel.setLeafSize(0.5f, 0.5f, 0.5f); // Only for publishing
                    publish_voxel.filter(*global_map_downsampled);
                    sensor_msgs::msg::PointCloud2 global_map_msg;
                    pcl::toROSMsg(*global_map_downsampled, global_map_msg);
                    global_map_msg.header.frame_id = "map";
                    global_map_msg.header.stamp = this->get_clock()->now();
                    global_map_pub_->publish(global_map_msg);
                    RCLCPP_INFO(get_logger(), "Published downsampled global map to /global_map");
                }
                response->success = true;
                response->message = "Map update successfully.";
                RCLCPP_INFO(get_logger(), "Global map updated!!!!!!");
                registration->setInputTarget(global_map_points_ptr_);;
                Reset();
                update_map_flag_.store(false);
            }
        }
        else {
            RCLCPP_ERROR(get_logger(), "Loaded map is empty: %s", map_path.c_str());
            response->success = false;
            response->message = "Loaded map is empty.";
            update_map_flag_.store(false);
            return;
        }
        update_map_flag_.store(false);
    }

    void Reset() {
        const bool have_seed = has_set_init_pose_;
        reset_tracking_state_locked(
          get_clock()->now(), have_seed, last_init_pos_, last_init_quat_);
        gl_once_gate_ = use_global_localization_init_;
        RCLCPP_INFO(get_logger(),
          "Localization state fully reset; local fused odometry continuity preserved");
    }

private:
  std::string robot_odom_frame_id;
  std::string odom_child_frame_id;
  std::string localization_odom_frame_id;
  bool send_tf_transforms;
  bool send_odom_base_transform_ = false;
  bool publish_bootstrap_map_to_odom_ = true;
  double local_odom_publish_rate_hz_ = 50.0;
  double tracking_min_position_stddev_ = 0.05;
  double tracking_min_orientation_stddev_deg_ = 2.0;
  double initializing_position_stddev_ = 0.50;
  double initializing_orientation_stddev_deg_ = 30.0;
  double degraded_position_stddev_ = 0.75;
  double degraded_orientation_stddev_deg_ = 45.0;

  bool use_imu;
  bool invert_acc;
  bool invert_gyro;
  bool enable_internal_odom_ukf_ = false;

  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr                         imu_sub;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr                       motor_odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr                 points_sub;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr                 globalmap_sub;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialpose_sub;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr occupancy_map_sub_;
  rclcpp::TimerBase::SharedPtr localization_lidar_info_timer_;
  rclcpp::TimerBase::SharedPtr odom_publish_timer_;
  rclcpp::TimerBase::SharedPtr local_odom_publish_timer_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr               pose_pub;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr               local_odom_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr         aligned_pub;
  rclcpp::Publisher<localization::msg::ScanMatchingStatus>::SharedPtr status_pub;
  rclcpp::Publisher<robots_dog_msgs::msg::Localization>::SharedPtr    localization_info_pub_;
  rclcpp::Publisher<robots_dog_msgs::msg::Localization>::SharedPtr    localization_health_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr         global_map_pub_;

  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

  // imu input buffer
  mutable std::mutex imu_data_mutex;
  std::vector<sensor_msgs::msg::Imu::ConstSharedPtr> imu_data;
  size_t imu_buffer_max_size_ = 4096;
  int64_t last_received_imu_stamp_ns_ = 0;
  int64_t last_main_ukf_imu_stamp_ns_ = 0;
  uint64_t imu_received_count_ = 0;
  uint64_t imu_integrated_count_ = 0;
  uint64_t imu_duplicate_count_ = 0;
  uint64_t imu_out_of_order_count_ = 0;
  uint64_t imu_buffer_overflow_count_ = 0;
  uint64_t imu_main_ukf_rejected_count_ = 0;
  uint64_t imu_future_retained_count_ = 0;
  
  // transformation matrices 
  Eigen::Matrix3f init_rotation_matrix_ = Eigen::Matrix3f::Identity();
  Eigen::Matrix4f gravity_transform_    = Eigen::Matrix4f::Identity();
  
  // Global localization and pose management
  std::shared_ptr<GlobalLocalization> global_localization_ptr_;
  // Pose caching and change detection
  Eigen::Vector3f last_init_pos_     = Eigen::Vector3f::Zero();
  Eigen::Quaternionf last_init_quat_ = Eigen::Quaternionf::Identity();
  bool has_set_init_pose_            = false;
  std::string last_pose_source_      = "none";
  // Configuration parameters for initial pose
  bool specify_init_pose_ = true;
  double init_pos_x_ = 0.0;
  double init_pos_y_ = 0.0;
  double init_pos_z_ = 0.0;
  double init_ori_w_ = 1.0;
  double init_ori_x_ = 0.0;
  double init_ori_y_ = 0.0;
  double init_ori_z_ = 0.0;
  // Global localization parameters
  bool use_global_localization_init_ = true;
  float init_pose_change_threshold_ = 0.01f;      
  float init_quat_change_threshold_ = 0.01f;      
  float global_localization_timeout_ = 10.0f;     
  // Global localization state
  bool global_localization_in_progress_ = false;
  rclcpp::Time global_localization_start_time_;
  bool gl_once_gate_ = true;
  // Sensor data validity tracking
  rclcpp::Time last_lidar_data_time_;
  rclcpp::Time last_imu_data_time_;
  std::deque<bool> lidar_status_buffer_;
  std::deque<bool> imu_status_buffer_;
  int min_valid_count_;      
  int buffer_size_;          
  double sensor_timeout_threshold_ = 1.0; 

  // Store previous state for extrapolation
  Eigen::Vector3f last_velocity_{0.0, 0.0, 0.0};           // Previous velocity
  Eigen::Vector3f last_angular_velocity_{0.0, 0.0, 0.0};   // Previous angular velocity
  Eigen::Matrix4f last_pose_{Eigen::Matrix4f::Identity()};  // Previous pose matrix
  Eigen::Matrix4f last_reliable_pose_{Eigen::Matrix4f::Identity()};
  bool has_reliable_pose_ = false;
  rclcpp::Time last_valid_pose_time_;                        // Time of last valid pose
  bool has_valid_pose_history_ = false;                      // Whether valid pose history exists
  
  // Store latest IMU data for extrapolation
  Eigen::Vector3f latest_angular_velocity_{0.0, 0.0, 0.0}; // Latest IMU angular velocity
  
  // Confidence management
  double current_confidence_ = 1.0;                    // Current confidence
  rclcpp::Time last_confidence_update_time_;           // Last confidence update time
  double confidence_decay_rate_ = 0.1;                 // Confidence decay rate (per second)
  bool is_extrapolating_ = false;                      // Whether currently extrapolating
  
  int log_counter_ = 0;                                // Log counter
  int log_interval_ = 10;                              // Log printing interval (print every 10 times)

  pcl::PointCloud<PointT>::Ptr globalmap;
  pcl::Registration<PointT, PointT>::Ptr registration;
  pcl::PointCloud<PointT>::Ptr global_map_points_ptr_;
  pcl::VoxelGrid<PointT>::Ptr  voxel_filter_ptr_ = pcl::VoxelGrid<PointT>::Ptr(new pcl::VoxelGrid<PointT>());
  pcl::PointCloud<PointT>::Ptr raw_points_ptr_ = nullptr;  ///< Raw point cloud pointer.
  // pose estimator
  std::mutex pose_estimator_mutex;
  std::unique_ptr<localization::PoseEstimator> pose_estimator;

  rclcpp::Service<robots_dog_msgs::srv::LocalizationState>::SharedPtr localization_state_srv_;
  rclcpp::Service<robots_dog_msgs::srv::LoadMap>::SharedPtr load_map_service_ptr_;
  // Parameters
  double cool_time_duration;
  std::string reg_method;
  std::string ndt_neighbor_search_method;
  double ndt_neighbor_search_radius;
  double ndt_resolution;
  bool enable_robot_odometry_prediction;
  int imu_data_filter_num_ = 5;  // Number of IMU data points to filter.

  bool is_init_success_ = false;
  bool is_use_map_coord_ = true;
  int localization_state_ = 0;   // 0: not init, 1: initing, 2: init success, 3: continuous localization, 4: continuous localization failed
  sensor_msgs::msg::Imu::SharedPtr correct_imu_data_ptr_;
  std::atomic<bool> update_map_flag_{false};
  float globalmap_voxel_size_ = 0.3;
  float points_voxel_filter_size_ = 0.30f;
  int last_timeout_;
  std::atomic<ModeState>  mode_state_ = ModeState::INIT;
};
}  // namespace localization

int main(int argc, char** argv) {
  omp_set_num_threads(6);
  rclcpp::init(argc, argv);
  auto node = std::make_shared<localization::HdlLocalizationNode>(rclcpp::NodeOptions());

  // Keep LiDAR/NDT, IMU and motor odometry callbacks independently serviceable.
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 3);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
