#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
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

    // Precipitation filter (off unless speckle_region is set): inside the region,
    // a point that is dim (intensity <= speckle_max_intensity) AND isolated (fewer
    // than speckle_min_neighbors other kept points within speckle_radius) is
    // dropped. Falling snow and drizzle close to the lidars look like that; real
    // surfaces at that range are dense (2026-11-18: snow 3-6 intensity, <= 3
    // neighbours in 0.1 m; a person next to the robot >= 10, hundreds).
    const auto region = declare_parameter<std::vector<double>>(
      "speckle_region", std::vector<double>{});
    if (!region.empty() && region.size() != 6) {
      throw std::runtime_error("speckle_region must be empty or six values");
    }
    for (size_t i = 0; i < region.size(); ++i) {
      speckle_region_[i] = static_cast<float>(region[i]);
    }
    speckle_enabled_ = region.size() == 6;
    speckle_radius_ = static_cast<float>(declare_parameter<double>("speckle_radius", 0.10));
    speckle_min_neighbors_ = static_cast<int>(declare_parameter<int>("speckle_min_neighbors", 4));
    speckle_max_intensity_ = static_cast<float>(
      declare_parameter<double>("speckle_max_intensity", 7.0));
    if (speckle_enabled_ && (!(speckle_radius_ > 0.0f) || speckle_min_neighbors_ < 1)) {
      throw std::runtime_error("speckle_radius must be > 0 and speckle_min_neighbors >= 1");
    }

    const auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic, qos);
    subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic, qos,
      std::bind(&PointCloudSelfFilter::filter, this, std::placeholders::_1));
    RCLCPP_INFO(
      get_logger(), "Point-cloud self-filter: %s -> %s, boxes=%zu",
      input_topic.c_str(), output_topic.c_str(), boxes_.size());
    if (speckle_enabled_) {
      RCLCPP_INFO(
        get_logger(),
        "Precipitation filter: region [%.2f %.2f %.2f %.2f %.2f %.2f], drop intensity <= %.1f "
        "with < %d neighbours within %.2f m",
        speckle_region_[0], speckle_region_[1], speckle_region_[2], speckle_region_[3],
        speckle_region_[4], speckle_region_[5], speckle_max_intensity_,
        speckle_min_neighbors_, speckle_radius_);
    }
  }

private:
  struct FieldOffsets {
    int x = -1;
    int y = -1;
    int z = -1;
    int intensity = -1;
  };

  struct Kept {
    size_t offset;
    float x, y, z, intensity;
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
      if (field.name == "intensity") result.intensity = static_cast<int>(field.offset);
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

    // Pass 1: drop non-finite points and the exclusion boxes.
    const bool speckle = speckle_enabled_ && fields.intensity >= 0 &&
      static_cast<size_t>(fields.intensity) + sizeof(float) <= cloud->point_step;
    if (speckle_enabled_ && !speckle) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 10000,
        "Precipitation filter needs a FLOAT32 intensity field; not applied");
    }
    std::vector<Kept> kept;
    kept.reserve(static_cast<size_t>(cloud->width) * cloud->height);
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
        float intensity = 0.0f;
        if (speckle) std::memcpy(&intensity, point + fields.intensity, sizeof(float));
        kept.push_back({point_offset, x, y, z, intensity});
      }
    }

    // Pass 2: precipitation speckle near the robot.
    std::vector<char> drop(kept.size(), 0);
    size_t speckle_removed = 0;
    if (speckle) speckle_removed = mark_speckle(kept, drop);

    for (size_t i = 0; i < kept.size(); ++i) {
      if (drop[i]) continue;
      const uint8_t* point = cloud->data.data() + kept[i].offset;
      output.data.insert(output.data.end(), point, point + cloud->point_step);
    }
    output.width = static_cast<uint32_t>(output.data.size() / output.point_step);
    output.row_step = output.width * output.point_step;
    publisher_->publish(output);
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Self-filter kept=%u removed=%zu precipitation=%zu", output.width, removed,
      speckle_removed);
  }

  bool in_region(const Kept& p, float margin) const {
    return p.x >= speckle_region_[0] - margin && p.x <= speckle_region_[1] + margin &&
           p.y >= speckle_region_[2] - margin && p.y <= speckle_region_[3] + margin &&
           p.z >= speckle_region_[4] - margin && p.z <= speckle_region_[5] + margin;
  }

  int64_t cell_key(int ix, int iy, int iz) const {
    // 21 bits per axis: plenty for a few-metre region at centimetre cells.
    constexpr int64_t kOffset = 1 << 20;
    return ((static_cast<int64_t>(ix) + kOffset) << 42) |
           ((static_cast<int64_t>(iy) + kOffset) << 21) |
           (static_cast<int64_t>(iz) + kOffset);
  }

  // Marks dim points inside the region with fewer than speckle_min_neighbors kept
  // neighbours within speckle_radius. Neighbours are searched among the kept points
  // inside the region grown by the radius (a hash grid with radius-sized cells).
  size_t mark_speckle(const std::vector<Kept>& kept, std::vector<char>& drop) const {
    const float r = speckle_radius_;
    const float r2 = r * r;
    std::unordered_map<int64_t, std::vector<uint32_t>> grid;
    std::vector<uint32_t> candidates;
    for (uint32_t i = 0; i < kept.size(); ++i) {
      const Kept& p = kept[i];
      if (!in_region(p, r)) continue;
      grid[cell_key(static_cast<int>(std::floor(p.x / r)), static_cast<int>(std::floor(p.y / r)),
                    static_cast<int>(std::floor(p.z / r)))].push_back(i);
      if (p.intensity <= speckle_max_intensity_ && in_region(p, 0.0f)) candidates.push_back(i);
    }
    size_t marked = 0;
    for (const uint32_t i : candidates) {
      const Kept& p = kept[i];
      const int cx = static_cast<int>(std::floor(p.x / r));
      const int cy = static_cast<int>(std::floor(p.y / r));
      const int cz = static_cast<int>(std::floor(p.z / r));
      int neighbours = 0;
      for (int dx = -1; dx <= 1 && neighbours < speckle_min_neighbors_; ++dx) {
        for (int dy = -1; dy <= 1 && neighbours < speckle_min_neighbors_; ++dy) {
          for (int dz = -1; dz <= 1 && neighbours < speckle_min_neighbors_; ++dz) {
            const auto it = grid.find(cell_key(cx + dx, cy + dy, cz + dz));
            if (it == grid.end()) continue;
            for (const uint32_t j : it->second) {
              if (j == i) continue;
              const float ex = kept[j].x - p.x;
              const float ey = kept[j].y - p.y;
              const float ez = kept[j].z - p.z;
              if (ex * ex + ey * ey + ez * ez <= r2 && ++neighbours >= speckle_min_neighbors_) break;
            }
          }
        }
      }
      if (neighbours < speckle_min_neighbors_) {
        drop[i] = 1;
        ++marked;
      }
    }
    return marked;
  }

  std::vector<std::array<float, 6>> boxes_;
  bool speckle_enabled_{false};
  std::array<float, 6> speckle_region_{};
  float speckle_radius_{0.10f};
  int speckle_min_neighbors_{4};
  float speckle_max_intensity_{7.0f};
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
