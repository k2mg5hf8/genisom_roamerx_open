#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

namespace localization {

class PointCloudSelfFilter : public rclcpp::Node {
public:
  explicit PointCloudSelfFilter(const rclcpp::NodeOptions& options)
  : Node("pointcloud_self_filter", options) {
    const auto input_topic = declare_parameter<std::string>("input_topic", "/front_lidar");
    const auto output_topic = declare_parameter<std::string>(
      "output_topic", "/front_lidar_filtered");
    const auto flat_boxes = declare_parameter<std::vector<double>>(
      "exclusion_boxes", std::vector<double>{});
    if (flat_boxes.size() % 6 != 0) {
      throw std::runtime_error("exclusion_boxes must contain groups of six values");
    }
    for (size_t offset = 0; offset < flat_boxes.size(); offset += 6) {
      boxes_.push_back({
        static_cast<float>(flat_boxes[offset]),
        static_cast<float>(flat_boxes[offset + 1]),
        static_cast<float>(flat_boxes[offset + 2]),
        static_cast<float>(flat_boxes[offset + 3]),
        static_cast<float>(flat_boxes[offset + 4]),
        static_cast<float>(flat_boxes[offset + 5])});
    }

    const auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic, qos);
    subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic, qos,
      std::bind(&PointCloudSelfFilter::filter, this, std::placeholders::_1));
    RCLCPP_INFO(
      get_logger(), "Point-cloud self-filter: %s -> %s, boxes=%zu",
      input_topic.c_str(), output_topic.c_str(), boxes_.size());
  }

private:
  struct FieldOffsets {
    int x = -1;
    int y = -1;
    int z = -1;
  };

  static FieldOffsets field_offsets(const sensor_msgs::msg::PointCloud2& cloud) {
    FieldOffsets result;
    for (const auto& field : cloud.fields) {
      if (field.datatype != sensor_msgs::msg::PointField::FLOAT32 || field.count != 1) {
        continue;
      }
      if (field.name == "x") result.x = static_cast<int>(field.offset);
      if (field.name == "y") result.y = static_cast<int>(field.offset);
      if (field.name == "z") result.z = static_cast<int>(field.offset);
    }
    return result;
  }

  bool excluded(float x, float y, float z) const {
    return std::any_of(boxes_.begin(), boxes_.end(), [&](const auto& box) {
      return x >= box[0] && x <= box[1] &&
             y >= box[2] && y <= box[3] &&
             z >= box[4] && z <= box[5];
    });
  }

  void filter(const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud) {
    const FieldOffsets fields = field_offsets(*cloud);
    if (fields.x < 0 || fields.y < 0 || fields.z < 0 || cloud->is_bigendian ||
        cloud->point_step == 0 ||
        static_cast<size_t>(std::max({fields.x, fields.y, fields.z})) + sizeof(float) >
          cloud->point_step) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Self-filter requires little-endian FLOAT32 x/y/z fields; publishing nothing");
      return;
    }

    sensor_msgs::msg::PointCloud2 output;
    output.header = cloud->header;
    output.height = 1;
    output.fields = cloud->fields;
    output.is_bigendian = cloud->is_bigendian;
    output.point_step = cloud->point_step;
    output.is_dense = cloud->is_dense;
    output.data.reserve(static_cast<size_t>(cloud->width) * cloud->height * cloud->point_step);

    size_t removed = 0;
    for (uint32_t row = 0; row < cloud->height; ++row) {
      for (uint32_t column = 0; column < cloud->width; ++column) {
        const size_t point_offset = static_cast<size_t>(row) * cloud->row_step +
          static_cast<size_t>(column) * cloud->point_step;
        if (point_offset + cloud->point_step > cloud->data.size()) continue;
        const uint8_t* point = cloud->data.data() + point_offset;
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        std::memcpy(&x, point + fields.x, sizeof(float));
        std::memcpy(&y, point + fields.y, sizeof(float));
        std::memcpy(&z, point + fields.z, sizeof(float));
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
            excluded(x, y, z)) {
          ++removed;
          continue;
        }
        output.data.insert(output.data.end(), point, point + cloud->point_step);
      }
    }
    output.width = static_cast<uint32_t>(output.data.size() / output.point_step);
    output.row_step = output.width * output.point_step;
    publisher_->publish(output);
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Self-filter kept=%u removed=%zu", output.width, removed);
  }

  std::vector<std::array<float, 6>> boxes_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
};

}  // namespace localization

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<localization::PointCloudSelfFilter>(
    rclcpp::NodeOptions{}));
  rclcpp::shutdown();
  return 0;
}
