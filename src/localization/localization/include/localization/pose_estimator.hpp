#ifndef POSE_ESTIMATOR_HPP
#define POSE_ESTIMATOR_HPP

#include <functional>
#include <memory>
#include <limits>
#include <string>
#include <boost/optional.hpp>

#include <rclcpp/rclcpp.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/registration.h>

#include <rclcpp/rclcpp.hpp>

// struct MatchResult{
//     bool  is_converged_;   ///< Indicates whether the matching operation converged.
//     float fitness_score_;  ///< The fitness score of the matching operation.
//   };

namespace kkl {
  namespace alg {
template<typename T, class System> class UnscentedKalmanFilterX;
  }
}

namespace localization {

class PoseSystem;
class OdomSystem;

/**
 * @brief scan matching-based pose estimator
 */
class PoseEstimator {
public:
  using PointT = pcl::PointXYZI;
  using CorrectionValidator = std::function<bool(const Eigen::Matrix4f&, float)>;

  struct MatchResult{
    bool  is_converged_ = false;  ///< Indicates whether the matching operation converged.
    float fitness_score_ = std::numeric_limits<float>::infinity();  ///< The fitness score of the matching operation.
  };

  // Scan-to-map agreement of the NDT result, one nearest-neighbour pass over
  // the aligned scan against the registration's target kd-tree.
  struct ScanMetrics {
    bool valid = false;
    uint32_t points = 0;
    // PCL getFitnessScore(): mean squared NN distance over all points.
    float mean_sq = std::numeric_limits<float>::infinity();
    // mean(min(d^2, tau^2)): people, cars, vegetation and returns beyond the
    // map edge cost at most tau^2 each instead of their full squared distance.
    float capped_sq = std::numeric_limits<float>::infinity();
    float inlier_fraction = 0.0f;  // share of points with d <= tau
    float inlier_rmse = std::numeric_limits<float>::infinity();
  };

  enum class FitnessMetric { kMeanSquared, kCappedSquared };

  struct CorrectionDiagnostics {
    bool accepted = false;
    std::string rejection_reason = "not_run";
    Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f candidate_pose = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f innovation = Eigen::Matrix4f::Identity();
    // Guess actually handed to NDT (initial_guess plus the terrain hint) and
    // the unprojected 6-DoF NDT result.
    Eigen::Matrix4f registration_guess = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f raw_transform = Eigen::Matrix4f::Identity();
    bool has_raw_transform = false;
    int iterations = -1;
    ScanMetrics metrics;
  };

  static ScanMetrics compute_scan_metrics(
    const pcl::PointCloud<PointT>& aligned,
    const pcl::Registration<PointT, PointT>& registration,
    float inlier_distance);

  /**
   * @brief constructor
   * @param registration        registration method
   * @param stamp               timestamp
   * @param pos                 initial position
   * @param quat                initial orientation
   * @param cool_time_duration  during "cool time", prediction is not performed
   */
  PoseEstimator(pcl::Registration<PointT, PointT>::Ptr& registration, const rclcpp::Time& stamp, const Eigen::Vector3f& pos, 
    const Eigen::Quaternionf& quat, double cool_time_duration, Eigen::Vector3d bias_acc = Eigen::Vector3d(0.0, 0.0, 0.0),
    Eigen::Vector3d bias_gyro = Eigen::Vector3d(0.0, 0.0, 0.0));
  ~PoseEstimator();

  /**
   * @brief predict
   * @param stamp    timestamp
   */
  void predict(const rclcpp::Time& stamp);

  /**
   * @brief predict
   * @param stamp    timestamp
   * @param acc      acceleration
   * @param gyro     angular velocity
   */
  void predict(const rclcpp::Time& stamp, const Eigen::Vector3f& acc, const Eigen::Vector3f& gyro);

  /**
   * @brief update the state of the odomety-based pose estimation
   */
  void predict_odom(const Eigen::Matrix4f& odom_delta);

  /**
   * @brief mark the start of a scan cycle before attempting an odometry lookup
   */
  void reset_odom_prediction();

  /**
   * @brief correct
   * @param cloud   input cloud
   * @return cloud aligned to the globalmap
   */
  pcl::PointCloud<PointT>::Ptr correct(
    const rclcpp::Time& stamp,
    const pcl::PointCloud<PointT>::ConstPtr& cloud,
    const CorrectionValidator& validator = CorrectionValidator{},
    const Eigen::Matrix4f* initial_guess_override = nullptr,
    bool commit_correction = true);

  /* getters */
  rclcpp::Time last_correction_time() const;

  Eigen::Vector3f pos() const;
  Eigen::Vector3f vel() const;
  Eigen::Quaternionf quat() const;
  Eigen::Matrix4f matrix() const;
  Eigen::Matrix<float, 6, 6> pose_covariance() const;
  bool state_is_finite() const;

  Eigen::Vector3f odom_pos() const;
  Eigen::Quaternionf odom_quat() const;
  Eigen::Matrix4f odom_matrix() const;

  const boost::optional<Eigen::Matrix4f>& wo_prediction_error() const;
  const boost::optional<Eigen::Matrix4f>& imu_prediction_error() const;
  const boost::optional<Eigen::Matrix4f>& odom_prediction_error() const;

  MatchResult GetMatchState() const; 
  const CorrectionDiagnostics& correction_diagnostics() const;
  Eigen::VectorXf GetCurrentUkfState(); 

  // IMU bias setters (used after static IMU initialization)
  void set_initial_biases(const Eigen::Vector3f& acc_bias, const Eigen::Vector3f& gyro_bias);
  void apply_planar_constraint(float z, float roll, float pitch, bool zero_velocity);

  void set_max_xy_jump(float v) { max_xy_jump_ = v; }
  void set_planar_correction(bool enabled) { planar_correction_enabled_ = enabled; }
  void set_correction_limits(float max_xy_jump, float max_yaw_jump, float max_fitness_score) {
    max_xy_jump_ = max_xy_jump;
    max_yaw_jump_ = max_yaw_jump;
    max_fitness_score_ = max_fitness_score;
  }
  // Which ScanMetrics value is the fitness compared with every score threshold.
  void set_fitness_metric(FitnessMetric metric, float inlier_distance) {
    fitness_metric_ = metric;
    inlier_distance_ = inlier_distance;
  }
  // 0 disables; otherwise a correction also needs this share of inliers.
  void set_min_inlier_fraction(float value) { min_inlier_fraction_ = value; }
  // Terrain hint for the next correct(): the NDT initial guess takes this z,
  // roll and pitch instead of the planar ones. The planar projection of the
  // result (published pose, gates, UKF) is unchanged.
  void set_registration_hint(bool valid, float z, float roll, float pitch) {
    hint_valid_ = valid;
    hint_z_ = z;
    hint_roll_ = roll;
    hint_pitch_ = pitch;
  }

private:
  FitnessMetric fitness_metric_ = FitnessMetric::kMeanSquared;
  float inlier_distance_ = 0.5f;
  float min_inlier_fraction_ = 0.0f;
  bool hint_valid_ = false;
  float hint_z_ = 0.0f;
  float hint_roll_ = 0.0f;
  float hint_pitch_ = 0.0f;
  float max_xy_jump_ = 5.0f;
  bool planar_correction_enabled_ = true;
  float max_yaw_jump_ = std::numeric_limits<float>::infinity();
  float max_fitness_score_ = 0.2f;
  rclcpp::Time init_stamp;             // when the estimator was initialized
  rclcpp::Time prev_stamp;             // when the estimator was updated last time
  rclcpp::Time last_correction_stamp;  // when the estimator performed the correction step
  double cool_time_duration;        //

  Eigen::MatrixXf process_noise;
  MatchResult     match_result_;
  CorrectionDiagnostics correction_diagnostics_;
  std::unique_ptr<kkl::alg::UnscentedKalmanFilterX<float, PoseSystem>> ukf;
  std::unique_ptr<kkl::alg::UnscentedKalmanFilterX<float, OdomSystem>> odom_ukf;
  bool odom_prediction_available_ = false;

  Eigen::Matrix4f last_observation;
  boost::optional<Eigen::Matrix4f> wo_pred_error;
  boost::optional<Eigen::Matrix4f> imu_pred_error;
  boost::optional<Eigen::Matrix4f> odom_pred_error;

  pcl::Registration<PointT, PointT>::Ptr registration;
  rclcpp::Logger logger_ = rclcpp::get_logger("pose_estimator");  ///< The logger instance.
  };

}  // namespace localization

#endif  // POSE_ESTIMATOR_HPP
