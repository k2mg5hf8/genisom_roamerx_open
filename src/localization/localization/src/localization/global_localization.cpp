#include "localization/global_localization.hpp"
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>

namespace localization {

    GlobalLocalization::GlobalLocalization()
    : init_check_count_(0), last_position_(Eigen::Vector3d::Zero()) {}

GlobalLocalization::~GlobalLocalization() {}

void GlobalLocalization::init(const Eigen::Matrix4d& initial_pose) {
    initial_pose_ = initial_pose;
}


std::vector<Eigen::Matrix4d> GlobalLocalization::generateCandidates(const Eigen::Matrix4d& initial_pose){
    Eigen::Vector3d init_pos = initial_pose.block<3,1>(0,3);
    Eigen::Matrix3d init_rot = initial_pose.block<3, 3>(0, 0);
    Eigen::Matrix2d init_rot_2d = initial_pose.block<2, 2>(0, 0);
    std::vector<double> deltas_x = {0.0, 0.2};
    std::vector<double> deltas_y = {-0.2, 0.0, 0.2};
    std::vector<double> deltas_w = {-15.0, 0.0, 15}; // degrees
    std::vector<Eigen::Matrix4d> candidates;
    for (const auto& dx : deltas_x){
        for (const auto& dy : deltas_y){
             for (const auto& dw : deltas_w){
                Eigen::Matrix4d candidate = Eigen::Matrix4d::Identity();
                Eigen::Vector2d offset_xy = init_rot_2d * Eigen::Vector2d(dx, dy);
                candidate.block<3, 1>(0, 3) = init_pos + Eigen::Vector3d(offset_xy.x(), offset_xy.y(), 0.0);
                Eigen::Matrix3d rot_yaw_delta;
                double d_rad = dw * M_PI / 180.0;
                rot_yaw_delta = Eigen::AngleAxisd(d_rad, Eigen::Vector3d::UnitZ());

                Eigen::Matrix3d new_rot = rot_yaw_delta * init_rot;
                candidate.block<3, 3>(0, 0) = new_rot;
                candidates.push_back(candidate);
             }
        }
    }
    return candidates;    
}


bool GlobalLocalization::recoveryLocalization(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& global_map,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& current_cloud,
    const Eigen::Matrix4d& initial_trans,  
    Eigen::Matrix4d& final_pose
)
{
    std::vector<Eigen::Matrix4d> candidates = generateCandidates(initial_trans);

    // Downsample the scan and build the target KD-tree ONCE, outside the loop.
    // Doing this per-candidate (as a naive loop over performGlobalLocalization
    // would) rebuilds the tree on the full global map 18 times - that's what
    // made a recovery sweep take ~70s and freeze the node. Reused here instead.
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
    voxel_filter.setInputCloud(current_cloud);
    voxel_filter.setLeafSize(0.2f, 0.2f, 0.2f);
    voxel_filter.filter(*current_cloud_filtered);

    pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
    icp.setInputTarget(global_map);
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);

    bool found_any = false;
    double best_score = 1000.0;  // fitness scores here are always well under 1.0
    Eigen::Matrix4d best_pose = Eigen::Matrix4d::Identity();

    for (auto& cand : candidates) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_guess(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::transformPointCloud(*current_cloud_filtered, *cloud_guess, cand.cast<float>());

        icp.setInputSource(cloud_guess);
        pcl::PointCloud<pcl::PointXYZI> aligned;
        icp.align(aligned);  // single pass here: enough to rank candidates cheaply

        double score = icp.getFitnessScore();
        if (score > 0.25) {
            continue;
        }

        Eigen::Matrix4d candidate_pose = icp.getFinalTransformation().cast<double>() * cand;

        if (score < best_score) {
            found_any = true;
            best_score = score;
            best_pose = candidate_pose;
        }

        if (best_score < 0.03) {
            break;  // already excellent, stop searching further candidates
        }
    }

    if (found_any) {
        final_pose = best_pose;
        return true;
    }
    return false;
}


bool GlobalLocalization::performGlobalLocalization(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& global_map,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& current_cloud,
    const Eigen::Matrix4d& initial_trans,  
    Eigen::Matrix4d& final_pose
) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud_ds(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>());

    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
    voxel_filter.setInputCloud(current_cloud);
    voxel_filter.setLeafSize(0.2f, 0.2f, 0.2f); 
    voxel_filter.filter(*current_cloud_filtered);

    pcl::transformPointCloud(*current_cloud_filtered, *current_cloud_ds, initial_trans.cast<float>());

    // ICP 
    pcl::IterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> icp;
    icp.setInputTarget(global_map);
    icp.setInputSource(current_cloud_ds);
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);

    pcl::PointCloud<pcl::PointXYZI> aligned_cloud;
    Eigen::Matrix4f T_corr_current;

    icp.align(aligned_cloud);
    T_corr_current = icp.getFinalTransformation();
    fitness_score = icp.getFitnessScore();
    RCLCPP_INFO(logger_, "ICP fitness score 1: %.6f", fitness_score);

    pcl::transformPointCloud(*current_cloud_ds, *current_cloud_ds, T_corr_current);


    if (fitness_score > 0.25) {
        return false; 
    }

    // 第二次 ICP 
    icp.align(aligned_cloud);
    Eigen::Matrix4f T_corr_second = icp.getFinalTransformation();
    fitness_score = icp.getFitnessScore();
    RCLCPP_INFO(logger_, "ICP fitness score 2: %.6f", fitness_score);
    pcl::transformPointCloud(*current_cloud_ds, *current_cloud_ds, T_corr_second);

    Eigen::Matrix4d T_corr_final = T_corr_second.cast<double>() * T_corr_current.cast<double>() * initial_trans;

    if (in_candidate_search_) {
        // Sweeping unrelated candidate poses in one frame: "matches the previous
        // call" would compare against the previous candidate, not a repeat of
        // this same hypothesis, so that check is meaningless here. Fitness alone
        // (already checked above) is the acceptance criterion.
        final_pose = T_corr_final;
        return true;
    }

    Eigen::Vector3d current_position = T_corr_final.block<3, 1>(0, 3);
    if (init_check_count_ > 0) {
        Eigen::Vector3d position_diff = current_position - last_position_;
        if (position_diff.norm() < 0.1) {
            init_check_count_++;
        } else {
            init_check_count_ = 0;
        }
    } else {
        init_check_count_ = 1;
    }

    last_position_ = current_position;

    if (init_check_count_ >= 2) {
        final_pose = T_corr_final;
        return true;
    }

    return false;
}
} // namespace localization
