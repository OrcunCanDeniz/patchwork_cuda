//
// Created by orcun on 10.05.2025.
//

#ifndef PATCHWORK_PATCHWORK_GPU_CUH
#define PATCHWORK_PATCHWORK_GPU_CUH

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <boost/format.hpp>
#include <jsk_recognition_msgs/PolygonArray.h>
#include <pcl/common/centroid.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include "zone_models_gpu.cuh"
#include "ros/ros.h"
#include "cuda_utils.cuh"

// vector cout operator
template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec) {
  os << "[ ";
  for (const auto &element : vec) {
    os << element << " ";
  }
  os << "]";
  return os;
}

// conditional parameter loading function for unorthodox (and normal) parameter
// locations
template <typename T>
bool condParam(ros::NodeHandle *nh,
               const std::string &param_name,
               T &param_val,
               const T &default_val,
               const std::string &prefix = "/patchwork") {
  if (nh->hasParam(prefix + "/" + param_name)) {
    if (nh->getParam(prefix + "/" + param_name, param_val)) {
      ROS_INFO_STREAM("param '" << prefix << "/" << param_name << "' -> '" << param_val << "'");
      return true;
    }
  } else if (nh->hasParam(ros::this_node::getName() + "/" + param_name)) {
    if (nh->getParam(ros::this_node::getName() + "/" + param_name, param_val)) {
      ROS_INFO_STREAM("param '" << ros::this_node::getName() << "/" << param_name << "' -> '"
                                << param_val << "'");
      return true;
    }
  }
  param_val = default_val;
  ROS_INFO_STREAM("param '" << param_name << "' -> '" << param_val << "' (default)");
  return false;
}

#define NUM_HEURISTIC_MAX_PTS_IN_PATCH 3000
#define MARKER_Z_VALUE -2.2

// Below colors are for visualization purpose
#define COLOR_CYAN 0.55                        // cyan
#define COLOR_GREEN 0.2                        // green
#define COLOR_BLUE 0.0                         // blue
#define COLOR_RED 1.0                          // red
#define COLOR_GLOBALLY_TOO_HIGH_ELEVATION 0.8  // I'm not sure...haha

int NOT_ASSIGNED = -2;
int FEW_POINTS = -1;
int UPRIGHT_ENOUGH = 0;      // cyan
int FLAT_ENOUGH = 1;         // green
int TOO_HIGH_ELEVATION = 2;  // blue
int TOO_TILTED = 3;          // red
int GLOBALLY_TOO_HIGH_ELEVATION = 4;

template <typename PointT>
struct Patch {
  bool is_close_to_origin_ = false;  // If so, we can set threshold more conservatively
  int ring_idx_ = NOT_ASSIGNED;
  int sector_idx_ = NOT_ASSIGNED;

  int status_ = NOT_ASSIGNED;

//  PCAFeature feature_;

  pcl::PointCloud<PointT> cloud_;
  pcl::PointCloud<PointT> ground_;
  pcl::PointCloud<PointT> non_ground_;

  void clear() {
    if (!cloud_.empty()) cloud_.clear();
    if (!ground_.empty()) ground_.clear();
    if (!non_ground_.empty()) non_ground_.clear();
  }
};

std::vector<float> COLOR_MAP = {COLOR_CYAN,
                                COLOR_GREEN,
                                COLOR_BLUE,
                                COLOR_RED,
                                COLOR_GLOBALLY_TOO_HIGH_ELEVATION};

template <typename PointT>
class PatchWorkGPU {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::vector<Patch<PointT>> Ring;
  typedef std::vector<Ring> RegionwisePatches;

  PatchWorkGPU() {}

  std::string frame_patchwork = "map";

  template <typename T>
  bool condParam(ros::NodeHandle *nh,
                 const std::string &param_name,
                 T &param_val,
                 const T &default_val,
                 const std::string &prefix = "/patchwork") const {
    if (nh->hasParam(prefix + "/" + param_name)) {
      if (nh->getParam(prefix + "/" + param_name, param_val)) {
        ROS_INFO_STREAM("param '" << prefix << "/" << param_name << "' -> '" << param_val << "'");
        return true;
      }
    } else if (nh->hasParam(ros::this_node::getName() + "/" + param_name)) {
      if (nh->getParam(ros::this_node::getName() + "/" + param_name, param_val)) {
        ROS_INFO_STREAM("param '" << ros::this_node::getName() << "/" << param_name << "' -> '"
                                  << param_val << "'");
        return true;
      }
    }
    param_val = default_val;
    ROS_INFO_STREAM("param '" << param_name << "' -> '" << param_val << "' (default)");
    return false;
  }

  explicit PatchWorkGPU(ros::NodeHandle *nh) {
    // Init ROS related
    ROS_INFO("Inititalizing PatchWork...");
    condParam(nh, "verbose", verbose_, false);
    condParam(nh, "sensor_height", sensor_height_, 1.723, "");
    condParam(nh, "sensor_model", sensor_model_, std::string("HDL-64E"), "");

    condParam(nh, "ATAT/ATAT_ON", ATAT_ON_, false);
    condParam(nh, "ATAT/max_r_for_ATAT", max_r_for_ATAT_, 5.0);
    condParam(nh, "ATAT/num_sectors_for_ATAT", num_sectors_for_ATAT_, 20);
    condParam(nh, "ATAT/noise_bound", noise_bound_, 0.2);

    condParam(nh, "num_iter", num_iter_, 3);
    condParam(nh, "num_lpr", num_lpr_, 20);
    condParam(nh, "num_min_pts", num_min_pts_, 10);
    condParam(nh, "th_seeds", th_seeds_, 0.4);
    condParam(nh, "th_dist", th_dist_, 0.3);
    condParam(nh, "max_r", max_range_, 80.0);
    condParam(nh, "min_r", min_range_,
              2.7);  // It should cover the body size of the car.
    condParam(nh, "uniform/num_rings", num_rings_, 30);
    condParam(nh, "uniform/num_sectors", num_sectors_, 108);
    condParam(nh, "uprightness_thr", uprightness_thr_,
              0.5);  // The larger, the more strict
    // The larger, the more soft
    condParam(nh, "adaptive_seed_selection_margin", adaptive_seed_selection_margin_, -1.1);

    // It is not in the paper
    // It is also not matched our philosophy, but it is employed to reject some
    // FPs easily & intuitively. For patchwork, it is only applied on Z3 and Z4
    condParam(nh, "using_global_elevation", using_global_thr_, true);
    condParam(nh, "global_elevation_threshold", global_elevation_thr_, 0.0);

    if (using_global_thr_) {
      std::cout << "\033[1;33m[Warning] Global elevation threshold is turned on :"
                << global_elevation_thr_ << "\033[0m" << std::endl;
    } else {
      std::cout << "Global thr. is not in use" << std::endl;
    }

    ROS_INFO("Sensor Height: %f", sensor_height_);
    ROS_INFO("Num of Iteration: %d", num_iter_);
    ROS_INFO("Num of LPR: %d", num_lpr_);
    ROS_INFO("Num of min. points: %d", num_min_pts_);
    ROS_INFO("Seeds Threshold: %f", th_seeds_);
    ROS_INFO("Distance Threshold: %f", th_dist_);
    ROS_INFO("Max. range:: %f", max_range_);
    ROS_INFO("Min. range:: %f", min_range_);
    ROS_INFO("adaptive_seed_selection_margin: %f", adaptive_seed_selection_margin_);

    // CZM denotes 'Concentric Zone Model'. Please refer to our paper
    // 2024.07.28. I feel `num_zones_`, `num_sectors_each_zone_`,
    // num_rings_each_zone_` are rarely fine-tuned. So I've decided to provide
    // predefined parameter sets for sensor types
    condParam(nh, "czm/elevation_thresholds", elevation_thr_, {0.523, 0.746, 0.879, 1.125});
    condParam(nh, "czm/flatness_thresholds", flatness_thr_, {0.0005, 0.000725, 0.001, 0.001});

    ROS_INFO("\033[1;32mUprightness\33[0m threshold: %f", uprightness_thr_);
    ROS_INFO("\033[1;32mElevation\33[0m thresholds: %f %f %f %f",
             elevation_thr_[0],
             elevation_thr_[1],
             elevation_thr_[2],
             elevation_thr_[3]);
    ROS_INFO("\033[1;32mFlatness\033[0m thresholds: %f %f %f %f",
             flatness_thr_[0],
             flatness_thr_[1],
             flatness_thr_[2],
             flatness_thr_[3]);
    ROS_INFO("Num. zones: %zu", zone_model_.num_zones_);
    condParam(nh, "max_pts_in_cloud", max_pts_in_cld_, 300000);

    // It equals to elevation_thr_.size()/flatness_thr_.size();
    zone_model_ = ConcentricZoneModel(sensor_model_, sensor_height_, min_range_, max_range_, max_pts_in_cld_);
    num_rings_of_interest_ = elevation_thr_.size();

    condParam(nh, "visualize", visualize_, true);
    condParam<std::string>(nh, "frame_patchwork", frame_patchwork, frame_patchwork);

    poly_list_.header.frame_id = frame_patchwork;
    poly_list_.polygons.reserve(130000);

    reverted_points_by_flatness_.reserve(NUM_HEURISTIC_MAX_PTS_IN_PATCH);

    PlanePub = nh->advertise<jsk_recognition_msgs::PolygonArray>("/gpf/plane", 100);
    RevertedCloudPub = nh->advertise<sensor_msgs::PointCloud2>("/revert_pc", 100);
    RejectedCloudPub = nh->advertise<sensor_msgs::PointCloud2>("/reject_pc", 100);

    const auto &num_sectors_each_zone_ = zone_model_.sensor_config_.num_sectors_for_each_zone_;
    sector_sizes_ = {2 * M_PI / num_sectors_each_zone_.at(0),
                     2 * M_PI / num_sectors_each_zone_.at(1),
                     2 * M_PI / num_sectors_each_zone_.at(2),
                     2 * M_PI / num_sectors_each_zone_.at(3)};
    std::cout << "INITIALIZATION COMPLETE" << std::endl;

    initialize(regionwise_patches_, zone_model_);

    CUDA_CHECK(cudaMalloc((void**)&cloud_in_d, sizeof(PointT) * max_pts_in_cld_));
    cudaStreamCreate(&stream);
  }

  void estimate_ground(const pcl::PointCloud<PointT> &cloud_in,
                       pcl::PointCloud<PointT> &ground,
                       pcl::PointCloud<PointT> &nonground,
                       double &time_taken);

  void toCUDA(const pcl::PointCloud<PointT> &cloud_in);


 private:
  cudaStream_t stream;
  uint32_t max_pts_in_cld_{300000};
  PointT *cloud_in_d{nullptr};

  // For ATAT (All-Terrain Automatic heighT estimator)
  bool ATAT_ON_;
  double noise_bound_;
  double max_r_for_ATAT_;
  int num_sectors_for_ATAT_;

  int num_iter_;
  int num_lpr_;
  int num_min_pts_;
  int num_rings_;
  int num_sectors_;
  int num_rings_of_interest_;

  double sensor_height_;
  double th_seeds_;
  double th_dist_;
  double max_range_;
  double min_range_;
  double uprightness_thr_;
  double adaptive_seed_selection_margin_;

  bool verbose_;
  bool initialized_ = true;

  // For global threshold
  bool using_global_thr_;
  double global_elevation_thr_;

  // For visualization
  bool visualize_;

  std::string sensor_model_;
  vector<std::pair<int, int>> patch_indices_;  // For multi-threading. {ring_idx, sector_idx}
  ConcentricZoneModel zone_model_;

  vector<double> sector_sizes_;
  vector<double> ring_sizes_;
  vector<double> min_ranges_;
  vector<double> elevation_thr_;
  vector<double> flatness_thr_;

  RegionwisePatches regionwise_patches_;

  jsk_recognition_msgs::PolygonArray poly_list_;

  ros::Publisher PlanePub, RevertedCloudPub, RejectedCloudPub;
  pcl::PointCloud<PointT> reverted_points_by_flatness_, rejected_points_by_elevation_;

  void initialize(RegionwisePatches &patches, const ConcentricZoneModel &zone_model);
};

template <typename PointT>
inline void PatchWorkGPU<PointT>::initialize(RegionwisePatches &patches,
                                             const ConcentricZoneModel &zone_model) {
  patches.clear();
  patch_indices_.clear();
  Patch<PointT> patch;

  // Reserve memory in advance to boost speed
  patch.cloud_.reserve(1000);
  patch.ground_.reserve(1000);
  patch.non_ground_.reserve(1000);

  // In polar coordinates, `num_columns` are `num_sectors`
  // and `num_rows` are `num_rings`, respectively
  int num_rows = zone_model_.num_total_rings_;
  const auto &num_sectors_per_ring = zone_model.num_sectors_per_ring_;

  for (int j = 0; j < num_rows; j++) {
    Ring ring;
    patch.ring_idx_ = j;
    patch.is_close_to_origin_ = j < zone_model.max_ring_index_in_first_zone;
    for (int i = 0; i < num_sectors_per_ring[j]; i++) {
      patch.sector_idx_ = i;
      ring.emplace_back(patch);

      patch_indices_.emplace_back(j, i);
    }
    patches.emplace_back(ring);
  }
}
#endif  // PATCHWORK_PATCHWORK_GPU_CUH
