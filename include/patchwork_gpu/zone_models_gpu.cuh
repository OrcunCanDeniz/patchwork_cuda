//
// Created by orcun on 10.05.2025.
//

#pragma once

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <numeric>

#include "patchwork/zone_models.hpp"
#include "cuda_utils.cuh"
#include "patchwork/point_type.hpp"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"


template<typename PointT>
class ConcentricZoneModelGPU: public ConcentricZoneModel
{
  public:
   std::size_t max_num_sectors_{0};
   std::size_t max_num_pts{0};
   std::vector<uint32_t> color_map;
   void  *cub_dev_scan_sum_tmp_ = nullptr;
   size_t cub_dev_scan_sum_tmp_bytes = 0;
   std::vector<uint> num_pts_per_patch_h;
   float* z_keys_d_ = nullptr;
   float* z_keys_out_d_ = nullptr;
   PointT* unsorted_patches_d_ = nullptr;
   void  *cub_sort_tmp_d = nullptr;
   size_t cub_sort_tmp_bytes = 0;
   PointMeta* metas_interm = nullptr;
   cudaStream_t czm_stream_;

  ConcentricZoneModelGPU(const std::string &sensor_model,
                        const double sensor_height,
                        const float min_range,
                        const float max_range,
                        const std::size_t max_num_points_in_pc)
      : ConcentricZoneModel(sensor_model, sensor_height, min_range, max_range),
          max_num_pts(max_num_points_in_pc)
  {
      std::transform(sensor_config_.num_laser_channels_per_zone_.begin(),
                      sensor_config_.num_laser_channels_per_zone_.end(),
                      std::back_inserter(sensor_config_.num_rings_for_each_zone_),
                     [](std::vector<int> &v) { return v.size();}
                    );

      max_num_sectors_ = *std::max_element(num_sectors_per_ring_.begin(),
                                           num_sectors_per_ring_.end());
      num_pts_per_patch_h.resize(num_total_rings_ * max_num_sectors_, 0);
      // Initialize the GPU constants
      cudaStreamCreateWithFlags(&czm_stream_, cudaStreamNonBlocking);
      set_cnst_mem();
      CUDA_CHECK(cudaMalloc((void**)&z_keys_d_, max_num_pts * sizeof(float)));
      CUDA_CHECK(cudaMalloc((void**)&z_keys_out_d_, max_num_pts * sizeof(float)));
      CUDA_CHECK(cudaMalloc((void**)&unsorted_patches_d_, max_num_pts * sizeof(PointT)));
      CUDA_CHECK(cudaMalloc((void**)&metas_interm, max_num_pts * sizeof(PointMeta)));


#ifdef VIZ
        constexpr uint8_t PALETTE[3][3] = {
            {255,   0,   0},  // red
            {  0, 255,   0},  // green
            {  0,   0, 255}   // blue
        };

        color_map.resize(num_total_rings_ * max_num_sectors_);
        for (size_t r = 0; r < num_total_rings_; ++r) {
          for (size_t s = 0; s < max_num_sectors_; ++s) {
            // choose 3-color index so that any neighbor (r±1,s) or (r,s±1) differs
            int idx = (static_cast<int>(r) + static_cast<int>(s)) % 3;
            auto &c = PALETTE[idx];
            uint32_t rgb = (uint32_t(c[0]) << 16)
                           | (uint32_t(c[1]) <<  8)
                           |  uint32_t(c[2]);
            color_map[r * max_num_sectors_ + s] = rgb;
          }
        }

#endif // VIZ

  }

  ConcentricZoneModelGPU() = default;

  ~ConcentricZoneModelGPU()
  {
    cudaFree(cub_dev_scan_sum_tmp_);
    cudaFree(z_keys_d_);
    cudaFree(z_keys_out_d_);
    cudaFree(unsorted_patches_d_);
  }

  void set_cnst_mem();

  bool create_patches_gpu(PointT* cloud_in_d, int num_pc_pts,
                          uint* num_pts_in_patch_d,
                          PointMeta* in_metas_d,
                          PointMeta* metas_d,
                          uint* offsets_d,
                          uint num_total_sectors,
                          PointT* patches_d,
                          uint& num_patched_pts_h,
                          cudaStream_t& stream);
};

