#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>

namespace localization {

class MotorOdomDeduplicator : public rclcpp::Node {
public:
  MotorOdomDeduplicator()
  : Node("motor_odom_deduplicator") {
    input_topic_ =
      declare_parameter<std::string>("input_topic", "/odom/mc_odom");
    const auto output_topic =
      declare_parameter<std::string>("output_topic", "/odom/mc_odom_unique");
    publisher_ = create_publisher<nav_msgs::msg::Odometry>(
      output_topic, rclcpp::QoS(rclcpp::KeepLast(16)).best_effort());
    subscription_ = create_subscription<nav_msgs::msg::Odometry>(
      input_topic_, rclcpp::QoS(rclcpp::KeepLast(16)).best_effort(),
      std::bind(&MotorOdomDeduplicator::callback, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(),
      "Publishing the first sample for each strictly increasing motor odometry stamp: %s -> %s",
      input_topic_.c_str(), output_topic.c_str());
  }

private:
  void callback(const nav_msgs::msg::Odometry::ConstSharedPtr message) {
    ++received_;

    const int64_t stamp_ns =
      rclcpp::Time(message->header.stamp, RCL_ROS_TIME).nanoseconds();
    if (has_last_stamp_ && stamp_ns <= last_stamp_ns_) {
      if (stamp_ns == last_stamp_ns_) {
        ++duplicates_;
      } else {
        ++out_of_order_;
      }
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Motor odom deduplication: received=%llu published=%llu duplicates=%llu "
        "out_of_order=%llu",
        static_cast<unsigned long long>(received_),
        static_cast<unsigned long long>(published_),
        static_cast<unsigned long long>(duplicates_),
        static_cast<unsigned long long>(out_of_order_));
      return;
    }

    last_stamp_ns_ = stamp_ns;
    has_last_stamp_ = true;
    ++published_;
    publisher_->publish(*message);
  }

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subscription_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr publisher_;
  std::string input_topic_;
  int64_t last_stamp_ns_ = 0;
  bool has_last_stamp_ = false;
  uint64_t received_ = 0;
  uint64_t published_ = 0;
  uint64_t duplicates_ = 0;
  uint64_t out_of_order_ = 0;
};

}  // namespace localization

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<localization::MotorOdomDeduplicator>());
  rclcpp::shutdown();
  return 0;
}
