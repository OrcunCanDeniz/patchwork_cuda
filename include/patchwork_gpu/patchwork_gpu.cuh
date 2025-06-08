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

#include <cusolverDn.h>

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)

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

enum PatchState : uint8_t
{
  FEW_PTS = 0,
  TOO_TILTED = 1,
  FLAT_ENOUGH = 2,
  TOO_HIGH_ELEV = 3,
  UPRIGHT_ENOUGH = 4,
  GLOB_TOO_HIGH_ELEV = 5
};

struct PCAFeature {
  float3 normal_;
  float3 singular_values_;
  float3 mean_;
  float d_;
  float th_dist_d_;
  float linearity_;
  float planarity_;
};

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
  void estimate_ground(pcl::PointCloud<PointT>* cloud_in,
                       pcl::PointCloud<PointT>* ground,
                       pcl::PointCloud<PointT>* nonground,
                       float* time_taken);
  void init_cuda();
  void setup_cusolver();
  void to_CUDA( pcl::PointCloud<PointT>* pc);
  void extract_init_seeds_gpu();
  void set_cnst_mem();
  void fit_regionwise_planes_gpu();
  void finalize_groundness_gpu();
  void to_pcl(pcl::PointCloud<PointT>* ground,
         pcl::PointCloud<PointT>* nonground,
         const bool with_lpr=false);

  void viz_points( pcl::PointCloud<PointT>* patched_pc,
                         pcl::PointCloud<PointT>* seed_pc );

  ~PatchWorkGPU()
  {
    // Safely reset GPU device only if CUDA context is valid
    if (cloud_in_d_) cudaFree(cloud_in_d_);

    if (patches_d) cudaFree(patches_d);

    if (metas_d) cudaFree(metas_d);

    if (num_pts_in_patch_d) cudaFree(num_pts_in_patch_d);

    if (patch_states_d)  cudaFree(patch_states_d);

    if (patch_offsets_h) cudaFreeHost(patch_offsets_h);

    if (patch_offsets_d) cudaFree(patch_offsets_d);

    if (cov_mats_d) cudaFree(cov_mats_d);

    if (pca_features_d) cudaFree(pca_features_d);

    if (eigen_vals_d) cudaFree(eigen_vals_d);

    if (eig_info_d) cudaFree(eig_info_d);

    if (num_patched_pts_h) cudaFreeHost(num_patched_pts_h);

    if (work_d) cudaFree(work_d);

    CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(syevj_params));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

#ifdef VIZ
    if (patches_h) {
      cudaFreeHost(patches_h);
      patches_h = nullptr;
    }

    if (num_pts_in_patch_h) {
      cudaFreeHost(num_pts_in_patch_h);
        num_pts_in_patch_h = nullptr;
    }

    if (metas_h) {
      cudaFreeHost(metas_h);
      metas_h = nullptr;
    }
#endif // VIZ
  }

  std::unique_ptr<ConcentricZoneModelGPU<PointT>> zone_model_;

  cudaStream_t stream_{nullptr};
  cudaStream_t streamd2h_{nullptr};
  cudaStream_t streamh2d_{nullptr};

  cudaEvent_t cuEvent_start, cuEvent_stop;

 private:
  long int max_pts_in_cld_{300000};
  PointT *cloud_in_d_{nullptr};

  PointT* patches_d;
  PointT* patches_h{nullptr};
  std::size_t patches_size{0};

  PointMeta* in_metas_d{nullptr};
  PointMeta* metas_d{nullptr};
  PointMeta* metas_h{nullptr};

  uint* num_pts_in_patch_d;
  uint* num_pts_in_patch_h{nullptr};
  std::size_t num_pts_in_patch_size{0};
  uint* num_patched_pts_h{nullptr};
  PointT* packed_pts_out_h{nullptr}; // first ground_pts_num elems is ground points

  PatchState* patch_states_d{nullptr};

  uint* patch_offsets_d{nullptr};  // For counting points in each patch
  uint* patch_offsets_h{nullptr};  // For counting points in each patch

  PCAFeature* pca_features_d{nullptr};
  double* cov_mats_d{nullptr};
  cusolverDnHandle_t cusolverH = NULL;
  syevjInfo_t syevj_params = NULL;
  double* eigen_vals_d{nullptr};
  double* work_d{nullptr};
  int* eig_info_d{nullptr};

  uint num_total_sectors_{0};
  uint last_sector_1st_ring_{0};

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
  float th_seeds_;
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
