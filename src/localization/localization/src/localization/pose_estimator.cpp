#include <localization/pose_estimator.hpp>

#include <algorithm>
#include <cmath>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <localization/pose_system.hpp>
#include <localization/odom_system.hpp>
#include <kkl/alg/unscented_kalman_filter.hpp>

namespace localization {
/**
 * @brief constructor
 * @param registration        registration method
 * @param stamp               timestamp
 * @param pos                 initial position
 * @param quat                initial orientation
 * @param cool_time_duration  during "cool time", prediction is not performed
 * @param bias_acc            initial acceleration bias
 * @param bias_gyro           initial gyro bias
 */
PoseEstimator::PoseEstimator(pcl::Registration<PointT, PointT>::Ptr& registration, const rclcpp::Time& stamp, 
    const Eigen::Vector3f& pos, const Eigen::Quaternionf& quat, double cool_time_duration, Eigen::Vector3d bias_acc, Eigen::Vector3d bias_gyro)
    : init_stamp(stamp), registration(registration), cool_time_duration(cool_time_duration) {

  prev_stamp = rclcpp::Time((int64_t)0, init_stamp.get_clock_type());
  last_correction_stamp = rclcpp::Time((int64_t)0, init_stamp.get_clock_type());
  last_observation = Eigen::Matrix4f::Identity();
  last_observation.block<3, 3>(0, 0) = quat.toRotationMatrix();
  last_observation.block<3, 1>(0, 3) = pos;

  process_noise = Eigen::MatrixXf::Identity(16, 16);
  process_noise.middleRows(0, 3) *= 0.5;     // 1.0
  process_noise.middleRows(3, 3) *= 1.0;
  process_noise.middleRows(6, 4) *= 0.5;
  process_noise.middleRows(10, 3) *= 1e-6;
  process_noise.middleRows(13, 3) *= 1e-6;

  Eigen::MatrixXf measurement_noise = Eigen::MatrixXf::Identity(7, 7);
  measurement_noise.middleRows(0, 3) *= 0.01;
  measurement_noise.middleRows(3, 4) *= 0.001;

  Eigen::VectorXf mean(16);
  mean.middleRows(0, 3) = pos;
  mean.middleRows(3, 3).setZero();
  mean.middleRows(6, 4) = Eigen::Vector4f(quat.w(), quat.x(), quat.y(), quat.z());
  mean.middleRows(10, 3) = bias_acc.cast<float>();
  mean.middleRows(13, 3) = bias_gyro.cast<float>();
  // mean.middleRows(13, 3).setZero();

  Eigen::MatrixXf cov = Eigen::MatrixXf::Identity(16, 16) * 0.01;

  PoseSystem system;
  ukf.reset(new kkl::alg::UnscentedKalmanFilterX<float, PoseSystem>(system, 16, 6, 7, process_noise, measurement_noise, mean, cov));
}

PoseEstimator::~PoseEstimator() {}

/**
 * @brief predict
 * @param stamp    timestamp
 * @param acc      acceleration
 * @param gyro     angular velocity
 */
void PoseEstimator::predict(const rclcpp::Time& stamp) {
  if ((stamp - init_stamp).seconds() < cool_time_duration || prev_stamp == rclcpp::Time((int64_t)0, prev_stamp.get_clock_type()) || prev_stamp == stamp) {
    prev_stamp = stamp;
    return;
  }

  double dt = (stamp - prev_stamp).seconds();
  prev_stamp = stamp;

  ukf->setProcessNoiseCov(process_noise * dt);
  ukf->system.dt = dt;

  ukf->predict();
}

/**
 * @brief predict
 * @param stamp    timestamp
 * @param acc      acceleration
 * @param gyro     angular velocity
 */
void PoseEstimator::predict(const rclcpp::Time& stamp, const Eigen::Vector3f& acc, const Eigen::Vector3f& gyro) {
  if (/*(stamp - init_stamp).seconds() < cool_time_duration || */prev_stamp == rclcpp::Time((int64_t)0, prev_stamp.get_clock_type()) || prev_stamp == stamp) {
    prev_stamp = stamp;
    RCLCPP_INFO(rclcpp::get_logger("PoseEstimator"), "Some ploblems with prev_stamp, not predict!");
    return;
  }

  double dt = (stamp - prev_stamp).seconds();
  if (dt > 0.1) {           
    prev_stamp = stamp;
    return;                 
  }
  if (dt > 0.05){
    dt = 0.05;
  } else if (dt < 0.0) {
    RCLCPP_INFO(rclcpp::get_logger("PoseEstimator"), "dt < 0.0, not predict!");
    return;
  }
  prev_stamp = stamp;

  ukf->setProcessNoiseCov(process_noise * dt);
  ukf->system.dt = dt;

  Eigen::VectorXf control(6);
  control.head<3>() = acc;
  control.tail<3>() = gyro;
  
  ukf->predict(control);
  
}

void PoseEstimator::set_initial_biases(const Eigen::Vector3f& acc_bias, const Eigen::Vector3f& gyro_bias){
  if(!ukf){
    return;
  }
  // state layout: [px,py,pz, vx,vy,vz, qw,qx,qy,qz, bax,bay,baz, bgx,bgy,bgz]
  ukf->mean.middleRows(10, 3) = acc_bias;
  ukf->mean.middleRows(13, 3) = gyro_bias;
}

void PoseEstimator::apply_planar_constraint(
    float z, float roll, float pitch, bool zero_velocity) {
  if (!ukf) {
    return;
  }

  const float yaw = std::atan2(
    matrix()(1, 0), matrix()(0, 0));
  const Eigen::Quaternionf attitude(
    Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) *
    Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *
    Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX()));

  ukf->mean(2) = z;
  ukf->mean(5) = 0.0f;
  if (zero_velocity) {
    ukf->mean.middleRows(3, 2).setZero();
  }
  ukf->mean.middleRows(6, 4) = Eigen::Vector4f(
    attitude.w(), attitude.x(), attitude.y(), attitude.z());
}

/**
 * @brief update the state of the odomety-based pose estimation
 */
void PoseEstimator::predict_odom(const Eigen::Matrix4f& odom_delta) {
  if(!odom_ukf) {
    Eigen::MatrixXf odom_process_noise = Eigen::MatrixXf::Identity(7, 7);
    Eigen::MatrixXf odom_measurement_noise = Eigen::MatrixXf::Identity(7, 7) * 1e-3;

    Eigen::VectorXf odom_mean(7);
    odom_mean.block<3, 1>(0, 0) = Eigen::Vector3f(ukf->mean[0], ukf->mean[1], ukf->mean[2]);
    odom_mean.block<4, 1>(3, 0) = Eigen::Vector4f(ukf->mean[6], ukf->mean[7], ukf->mean[8], ukf->mean[9]);
    Eigen::MatrixXf odom_cov = Eigen::MatrixXf::Identity(7, 7) * 1e-2;

    OdomSystem odom_system;
    odom_ukf.reset(new kkl::alg::UnscentedKalmanFilterX<float, OdomSystem>(odom_system, 7, 7, 7, odom_process_noise, odom_measurement_noise, odom_mean, odom_cov));
    // The main UKF has already been predicted to this scan timestamp. Applying
    // odom_delta here would count the first inter-scan motion twice.
    odom_prediction_available_ = false;
    return;
  }

  // invert quaternion if the rotation axis is flipped
  Eigen::Quaternionf quat(odom_delta.block<3, 3>(0, 0));
  if(odom_quat().coeffs().dot(quat.coeffs()) < 0.0) {
    quat.coeffs() *= -1.0f;
  }

  Eigen::VectorXf control(7);
  control.middleRows(0, 3) = odom_delta.block<3, 1>(0, 3);
  control.middleRows(3, 4) = Eigen::Vector4f(quat.w(), quat.x(), quat.y(), quat.z());

  Eigen::MatrixXf process_noise = Eigen::MatrixXf::Identity(7, 7);
  process_noise.topLeftCorner(3, 3) = Eigen::Matrix3f::Identity() * odom_delta.block<3, 1>(0, 3).norm() + Eigen::Matrix3f::Identity() * 1e-3;
  process_noise.bottomRightCorner(4, 4) = Eigen::Matrix4f::Identity() * (1 - std::abs(quat.w())) + Eigen::Matrix4f::Identity() * 1e-3;

  odom_ukf->setProcessNoiseCov(process_noise);
  odom_ukf->predict(control);
  odom_prediction_available_ = true;
}

void PoseEstimator::reset_odom_prediction() {
  odom_prediction_available_ = false;
}

/**
 * @brief correct
 * @param cloud   input cloud
 * @return cloud aligned to the globalmap
 */
pcl::PointCloud<PoseEstimator::PointT>::Ptr PoseEstimator::correct(
  const rclcpp::Time& stamp,
  const pcl::PointCloud<PointT>::ConstPtr& cloud,
  const CorrectionValidator& validator,
  const Eigen::Matrix4f* initial_guess_override,
  bool commit_correction) {
  last_correction_stamp = stamp;

  Eigen::Matrix4f init_guess = matrix();
  // Eigen::Matrix4f no_guess = last_observation;
  Eigen::Matrix4f imu_guess;
  Eigen::Matrix4f odom_guess = Eigen::Matrix4f::Identity();
  if (odom_ukf && odom_prediction_available_) {
    odom_guess = odom_matrix();
    // Motor translation is useful, but the robot's absolute motor yaw is
    // quantized and can jump by tens of degrees. Always retain the smooth
    // IMU-UKF orientation for the scan-matching initial guess.
    odom_guess.block<3, 3>(0, 0) = matrix().block<3, 3>(0, 0);
    init_guess = odom_guess;
  }
  if (initial_guess_override) {
    init_guess = *initial_guess_override;
  }
  correction_diagnostics_ = CorrectionDiagnostics{};
  correction_diagnostics_.initial_guess = init_guess;
  // Eigen::Matrix4f init_guess = Eigen::Matrix4f::Identity();

  // if(!odom_ukf) {
  //   init_guess = imu_guess = matrix();
  // } else {
  //   imu_guess = matrix();
  //   odom_guess = odom_matrix();

  //   Eigen::VectorXf imu_mean(7);
  //   Eigen::MatrixXf imu_cov = Eigen::MatrixXf::Identity(7, 7);
  //   imu_mean.block<3, 1>(0, 0) = ukf->mean.block<3, 1>(0, 0);
  //   imu_mean.block<4, 1>(3, 0) = ukf->mean.block<4, 1>(6, 0);

  //   imu_cov.block<3, 3>(0, 0) = ukf->cov.block<3, 3>(0, 0);
  //   imu_cov.block<3, 4>(0, 3) = ukf->cov.block<3, 4>(0, 6);
  //   imu_cov.block<4, 3>(3, 0) = ukf->cov.block<4, 3>(6, 0);
  //   imu_cov.block<4, 4>(3, 3) = ukf->cov.block<4, 4>(6, 6);

  //   Eigen::VectorXf odom_mean = odom_ukf->mean;
  //   Eigen::MatrixXf odom_cov = odom_ukf->cov;

  //   if (imu_mean.tail<4>().dot(odom_mean.tail<4>()) < 0.0) {
  //     odom_mean.tail<4>() *= -1.0;
  //   }

  //   Eigen::MatrixXf inv_imu_cov = imu_cov.inverse();
  //   Eigen::MatrixXf inv_odom_cov = odom_cov.inverse();

  //   Eigen::MatrixXf fused_cov = (inv_imu_cov + inv_odom_cov).inverse();
  //   Eigen::VectorXf fused_mean = fused_cov * inv_imu_cov * imu_mean + fused_cov * inv_odom_cov * odom_mean;

  //   init_guess.block<3, 1>(0, 3) = Eigen::Vector3f(fused_mean[0], fused_mean[1], fused_mean[2]);
  //   init_guess.block<3, 3>(0, 0) = Eigen::Quaternionf(fused_mean[3], fused_mean[4], fused_mean[5], fused_mean[6]).normalized().toRotationMatrix();
  // }

  pcl::PointCloud<PointT>::Ptr aligned(new pcl::PointCloud<PointT>());
  registration->setInputSource(cloud);
  registration->align(*aligned, init_guess);
  double ndt_score = registration->getFitnessScore();

  Eigen::Matrix4f trans = registration->getFinalTransformation();
  Eigen::Vector3f p = trans.block<3, 1>(0, 3);
  Eigen::Quaternionf q(trans.block<3, 3>(0, 0));

  if (planar_correction_enabled_) {
    const Eigen::Matrix3f initial_rotation = init_guess.block<3, 3>(0, 0);
    const float initial_roll = std::atan2(
      initial_rotation(2, 1), initial_rotation(2, 2));
    const float initial_pitch = std::asin(std::clamp(
      -initial_rotation(2, 0), -1.0f, 1.0f));
    const float candidate_yaw = std::atan2(trans(1, 0), trans(0, 0));
    q = Eigen::Quaternionf(
      Eigen::AngleAxisf(candidate_yaw, Eigen::Vector3f::UnitZ()) *
      Eigen::AngleAxisf(initial_pitch, Eigen::Vector3f::UnitY()) *
      Eigen::AngleAxisf(initial_roll, Eigen::Vector3f::UnitX()));
    p.z() = init_guess(2, 3);
    trans.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    trans.block<3, 1>(0, 3) = p;
  }
  correction_diagnostics_.candidate_pose = trans;
  correction_diagnostics_.innovation = init_guess.inverse() * trans;

  if(quat().coeffs().dot(q.coeffs()) < 0.0f) {
    q.coeffs() *= -1.0f;
  }

  float xy_jump = (p.head<2>() - init_guess.block<2, 1>(0, 3)).norm();
  Eigen::Matrix4f correction = init_guess.inverse() * trans;
  float yaw_jump = std::abs(std::atan2(correction(1, 0), correction(0, 0)));
  bool converged = registration->hasConverged();
  bool valid_score = std::isfinite(ndt_score) && ndt_score < max_fitness_score_;

  if (!converged || !valid_score || xy_jump > max_xy_jump_ || yaw_jump > max_yaw_jump_) {
    if (!converged) {
      correction_diagnostics_.rejection_reason = "not_converged";
    } else if (!std::isfinite(ndt_score)) {
      correction_diagnostics_.rejection_reason = "non_finite_fitness";
    } else if (!valid_score) {
      correction_diagnostics_.rejection_reason = "fitness_gate";
    } else if (xy_jump > max_xy_jump_) {
      correction_diagnostics_.rejection_reason = "xy_jump_gate";
    } else {
      correction_diagnostics_.rejection_reason = "yaw_jump_gate";
    }
    RCLCPP_WARN(rclcpp::get_logger("PoseEstimator"),
      "Rejecting NDT correction: converged=%d score=%.4f/%.4f xy=%.3f/%.3f m yaw=%.2f/%.2f deg",
      converged, ndt_score, max_fitness_score_, xy_jump, max_xy_jump_,
      yaw_jump * 180.0f / static_cast<float>(M_PI),
      max_yaw_jump_ * 180.0f / static_cast<float>(M_PI));
    match_result_.is_converged_ = false;
    match_result_.fitness_score_ = std::isfinite(ndt_score) ?
      static_cast<float>(ndt_score) : std::numeric_limits<float>::infinity();
    pcl::transformPointCloud(*cloud, *aligned, init_guess);
    odom_prediction_available_ = false;
    return aligned;
  }

  float max_delta_z = 0.15;
  if (!planar_correction_enabled_ && std::abs(p.z() - init_guess(2,3)) > max_delta_z ){
    RCLCPP_WARN(rclcpp::get_logger("PoseEstimator"), "Z position change too large: %.3f m, limiting to %.3f m", std::abs(p.z() - init_guess(2,3)), max_delta_z);
    p.z() = init_guess(2,3) + (p.z() > init_guess(2,3) ? max_delta_z : -max_delta_z);
  }

  if (!planar_correction_enabled_ && p(2) < -0.2 ) {
    RCLCPP_WARN(rclcpp::get_logger("PoseEstimator"), "Z position out of bounds: %f", p(2));
    p(2) =  -0.2;
  }
  if (!planar_correction_enabled_ && p(2) > 1.0 ) {
    RCLCPP_WARN(rclcpp::get_logger("PoseEstimator"), "Z position out of bounds: %f", p(2));
    p(2) =  1.0;
  }

  Eigen::Matrix4f candidate_pose = Eigen::Matrix4f::Identity();
  candidate_pose.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
  candidate_pose.block<3, 1>(0, 3) = p;
  correction_diagnostics_.candidate_pose = candidate_pose;
  correction_diagnostics_.innovation = init_guess.inverse() * candidate_pose;
  if (validator && !validator(candidate_pose, static_cast<float>(ndt_score))) {
    correction_diagnostics_.rejection_reason = "external_gate";
    match_result_.is_converged_ = false;
    match_result_.fitness_score_ = static_cast<float>(ndt_score);
    pcl::transformPointCloud(*cloud, *aligned, init_guess);
    odom_prediction_available_ = false;
    return aligned;
  }

  match_result_.is_converged_ = true;
  match_result_.fitness_score_ = static_cast<float>(ndt_score);
  correction_diagnostics_.accepted = true;
  correction_diagnostics_.rejection_reason.clear();

  if (!commit_correction) {
    odom_prediction_available_ = false;
    return aligned;
  }
  
  Eigen::VectorXf observation(7);
  observation.middleRows(0, 3) = p;
  observation.middleRows(3, 4) = Eigen::Vector4f(q.w(), q.x(), q.y(), q.z());
  last_observation = candidate_pose;

  // wo_pred_error = no_guess.inverse() * registration->getFinalTransformation();

  ukf->correct(observation);
  ukf->mean[5] = 0.0f; 
  // imu_pred_error = imu_guess.inverse() * registration->getFinalTransformation();

  if(odom_ukf) {
    if (observation.tail<4>().dot(odom_ukf->mean.tail<4>()) < 0.0) {
      odom_ukf->mean.tail<4>() *= -1.0;
    }

    odom_ukf->correct(observation);
    if (odom_prediction_available_) {
      odom_pred_error = odom_guess.inverse() * candidate_pose;
    } else {
      odom_pred_error = boost::none;
    }
  }
  odom_prediction_available_ = false;

  return aligned;
}

/* getters */
rclcpp::Time PoseEstimator::last_correction_time() const {
  return last_correction_stamp;
}

Eigen::Vector3f PoseEstimator::pos() const {
  return Eigen::Vector3f(ukf->mean[0], ukf->mean[1], ukf->mean[2]);
}

Eigen::Vector3f PoseEstimator::vel() const {
  return Eigen::Vector3f(ukf->mean[3], ukf->mean[4], ukf->mean[5]);
}

Eigen::Quaternionf PoseEstimator::quat() const {
  return Eigen::Quaternionf(ukf->mean[6], ukf->mean[7], ukf->mean[8], ukf->mean[9]).normalized();
}

Eigen::Matrix4f PoseEstimator::matrix() const {
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  m.block<3, 3>(0, 0) = quat().toRotationMatrix();
  m.block<3, 1>(0, 3) = pos();
  return m;
}

Eigen::Matrix<float, 6, 6> PoseEstimator::pose_covariance() const {
  Eigen::Matrix<float, 6, 6> result =
    Eigen::Matrix<float, 6, 6>::Zero();
  if (!ukf) {
    result.diagonal().setConstant(1e6f);
    return result;
  }

  const Eigen::MatrixXf& state_cov = ukf->getCov();
  if (state_cov.rows() < 10 || state_cov.cols() < 10) {
    result.diagonal().setConstant(1e6f);
    return result;
  }

  result.topLeftCorner<3, 3>() = state_cov.block<3, 3>(0, 0);

  const auto quat_to_rpy = [](Eigen::Vector4f qv) {
    Eigen::Quaternionf q(qv(0), qv(1), qv(2), qv(3));
    q.normalize();
    const float sinr = 2.0f * (q.w() * q.x() + q.y() * q.z());
    const float cosr = 1.0f - 2.0f * (q.x() * q.x() + q.y() * q.y());
    const float sinp = std::clamp(
      2.0f * (q.w() * q.y() - q.z() * q.x()), -1.0f, 1.0f);
    const float siny = 2.0f * (q.w() * q.z() + q.x() * q.y());
    const float cosy = 1.0f - 2.0f * (q.y() * q.y() + q.z() * q.z());
    return Eigen::Vector3f(
      std::atan2(sinr, cosr), std::asin(sinp), std::atan2(siny, cosy));
  };
  const auto wrapped_delta = [](float a, float b) {
    return std::atan2(std::sin(a - b), std::cos(a - b));
  };

  const Eigen::Vector4f qmean(
    ukf->mean(6), ukf->mean(7), ukf->mean(8), ukf->mean(9));
  Eigen::Matrix<float, 3, 4> jacobian;
  constexpr float epsilon = 1e-4f;
  for (int i = 0; i < 4; ++i) {
    Eigen::Vector4f plus = qmean;
    Eigen::Vector4f minus = qmean;
    plus(i) += epsilon;
    minus(i) -= epsilon;
    const Eigen::Vector3f rpy_plus = quat_to_rpy(plus);
    const Eigen::Vector3f rpy_minus = quat_to_rpy(minus);
    for (int axis = 0; axis < 3; ++axis) {
      jacobian(axis, i) =
        wrapped_delta(rpy_plus(axis), rpy_minus(axis)) / (2.0f * epsilon);
    }
  }

  const Eigen::Matrix<float, 4, 4> quat_cov = state_cov.block<4, 4>(6, 6);
  const Eigen::Matrix<float, 3, 4> position_quat_cov =
    state_cov.block<3, 4>(0, 6);
  result.bottomRightCorner<3, 3>() = jacobian * quat_cov * jacobian.transpose();
  result.topRightCorner<3, 3>() = position_quat_cov * jacobian.transpose();
  result.bottomLeftCorner<3, 3>() = result.topRightCorner<3, 3>().transpose();
  result = 0.5f * (result + result.transpose());

  for (int row = 0; row < 6; ++row) {
    for (int col = 0; col < 6; ++col) {
      if (!std::isfinite(result(row, col))) result(row, col) = 0.0f;
    }
    result(row, row) = std::clamp(result(row, row), 1e-8f, 1e6f);
  }
  return result;
}

Eigen::Vector3f PoseEstimator::odom_pos() const {
  return Eigen::Vector3f(odom_ukf->mean[0], odom_ukf->mean[1], odom_ukf->mean[2]);
}

Eigen::Quaternionf PoseEstimator::odom_quat() const {
  return Eigen::Quaternionf(odom_ukf->mean[3], odom_ukf->mean[4], odom_ukf->mean[5], odom_ukf->mean[6]).normalized();
}

Eigen::Matrix4f PoseEstimator::odom_matrix() const {
  Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
  m.block<3, 3>(0, 0) = odom_quat().toRotationMatrix();
  m.block<3, 1>(0, 3) = odom_pos();
  return m;
}

const boost::optional<Eigen::Matrix4f>& PoseEstimator::wo_prediction_error() const {
  return wo_pred_error;
}

const boost::optional<Eigen::Matrix4f>& PoseEstimator::imu_prediction_error() const {
  return imu_pred_error;
}

const boost::optional<Eigen::Matrix4f>& PoseEstimator::odom_prediction_error() const {
  return odom_pred_error;
}

PoseEstimator::MatchResult PoseEstimator::GetMatchState() const {
    return match_result_;
}

const PoseEstimator::CorrectionDiagnostics& PoseEstimator::correction_diagnostics() const {
  return correction_diagnostics_;
}

Eigen::VectorXf PoseEstimator::GetCurrentUkfState() {
    Eigen::Matrix4f state_matrix = matrix();
    Eigen::Vector3f t_state = state_matrix.block<3, 1>(0, 3);
    Eigen::Matrix3f rot = state_matrix.block<3, 3>(0, 0);
    Eigen::Quaternionf q_state(rot);
    q_state.normalize();
    Eigen::VectorXf state(7);
    state.head<3>() = t_state;
    state.tail<4>() << q_state.w(), q_state.x(), q_state.y(), q_state.z();
    return state;
}

}
