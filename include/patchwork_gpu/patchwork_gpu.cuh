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

template <typename PointT>
class PatchWorkGPU {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

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

  explicit PatchWorkGPU(ros::NodeHandle *nh);

  void reset_buffers(cudaStream_t stream=nullptr);
  void estimate_ground(pcl::PointCloud<PointT>* cloud_in);
  void init_cuda();
  void to_CUDA( pcl::PointCloud<PointT>* pc, cudaStream_t stream=0);
  uint32_t cuda_patches_to_pcl( pcl::PointCloud<PointT>* pc);



  ~PatchWorkGPU()
  {
    // Safely reset GPU device only if CUDA context is valid
    cudaFree(patches_d);  // only free device memory
    cudaFree(num_pts_in_patch_d);
    cudaFree(patch_offsets_d);

    if (cloud_in_d_) {
      cudaFree(cloud_in_d_);
    }

#ifdef VIZ_PATCHES
    if (patches_h) {
      cudaFreeHost(patches_h);
      patches_h = nullptr;
    }

    if (num_pts_in_patch_h) {
      cudaFreeHost(num_pts_in_patch_h);
        num_pts_in_patch_h = nullptr;
    }
#endif // VIZ_PATCHES
  }

  std::unique_ptr<ConcentricZoneModelGPU<PointT>> zone_model_;

  cudaStream_t stream_{nullptr};
  cudaStream_t streamd2h_{nullptr};
  cudaStream_t streamh2d_{nullptr};

 private:
  long int max_pts_in_cld_{300000};
  PointT *cloud_in_d_{nullptr};

  float4* patches_d;
  float4* patches_h{nullptr};
  std::size_t patches_size{0};

  PointMeta* metas_d{nullptr};

  uint* num_pts_in_patch_d;
  uint* num_pts_in_patch_h{nullptr};
  std::size_t num_pts_in_patch_size{0};

  uint* patch_offsets_d{nullptr};  // For counting points in each patch
  uint* patch_offsets_h{nullptr};  // For counting points in each patch

  cudaPitchedPtr lbr_d; // LPR = low point representative
  std::size_t lbr_size;
  uint num_total_sectors_{0};

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

  vector<double> sector_sizes_;
  vector<double> ring_sizes_;
  vector<double> min_ranges_;
  vector<double> elevation_thr_;
  vector<double> flatness_thr_;

  jsk_recognition_msgs::PolygonArray poly_list_;

  ros::Publisher PlanePub, RevertedCloudPub, RejectedCloudPub;
  pcl::PointCloud<PointT> reverted_points_by_flatness_, rejected_points_by_elevation_;

};


#endif  // PATCHWORK_PATCHWORK_GPU_CUH
