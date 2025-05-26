//
// Created by orcun on 10.05.2025.
//

#pragma once

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "patchwork/zone_models.hpp"
#include "patchwork/point_type.hpp"
#include "cuda_utils.cuh"
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
      // Initialize the GPU constants
      set_cnst_mem();

#ifdef VIZ_PATCHES
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

#endif // VIZ_PATCHES

  }

  ConcentricZoneModelGPU() = default;

  ~ConcentricZoneModelGPU()
  {
    cudaFree(cub_dev_scan_sum_tmp_);
  }

  void set_cnst_mem();

  bool create_patches_gpu(PointT* cloud_in_d, int num_pc_pts,
                          uint* num_pts_in_patch_d,
                          PointMeta* metas_d,
                          uint* offsets_d,
                          uint num_total_sectors,
                          float4* patches_d,
                          cudaStream_t& stream);
};

