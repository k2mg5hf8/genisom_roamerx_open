std_msgs/Header header

bool has_converged
bool accepted
float32 matching_error
float32 inlier_fraction
geometry_msgs/Transform relative_pose
geometry_msgs/Transform initial_guess
geometry_msgs/Transform candidate_pose
geometry_msgs/Transform innovation
string rejection_reason

float64 scan_age_sec
float64 imu_prediction_age_sec
float64 odom_prediction_age_sec
uint32 input_points
uint32 filtered_points
bool deskew_applied
uint32 deskewed_points
float64 scan_duration_sec
uint64 imu_received
uint64 imu_integrated
uint64 imu_duplicates
uint64 imu_out_of_order
uint64 imu_buffer_overflow
uint64 motor_odom_received
uint64 motor_odom_duplicates
uint64 motor_odom_out_of_order
float32 downsample_ms
float32 prediction_ms
float32 registration_ms
float32 total_callback_ms

std_msgs/String[] prediction_labels
geometry_msgs/Transform[] prediction_errors

# Scan metrics from one nearest-neighbour pass over the NDT-aligned scan
# (matching_error is the one named by fitness_metric; inlier_fraction uses
# all points at inlier_distance).
string fitness_metric
float32 mean_sq_error
float32 capped_sq_error
float32 inlier_distance
float32 inlier_rmse
uint32 metric_points
int32 ndt_iterations
# NDT initial guess (planar guess plus terrain z/roll/pitch when
# terrain_hint_used) and the unprojected 6-DoF NDT result.
bool terrain_hint_used
geometry_msgs/Transform registration_guess
geometry_msgs/Transform raw_transform
