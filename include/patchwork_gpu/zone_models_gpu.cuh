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

#include "cuda_utils.cuh"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

extern __device__ __constant__ float cnst_sqr_boundary_ranges[256];
extern __device__ __constant__ std::size_t cnst_boundary_ranges_size;
extern __device__ __constant__ float cnst_sqr_max_range;
extern __device__ __constant__ int cnst_num_sectors_per_ring[256];

template<typename PointT>
class ConcentricZoneModelGPU: public ConcentricZoneModel
{
  std::size_t max_num_sectors_{0};
  std::size_t max_num_pts{0};

  cudaPitchedPtr patches_d;
  char* patches_h{nullptr};
  std::size_t patches_size{0};

  cudaPitchedPtr num_pts_in_patch_d;
  char* num_pts_in_patch_h{nullptr};
  std::size_t num_pts_in_patch_size{0};

  PointT* cloud_in_d_{nullptr};

  std::vector<uint32_t> color_map;

  public:
  ConcentricZoneModelGPU(const std::string &sensor_model,
                        const double sensor_height,
                        const float min_range,
                        const float max_range,
                        const std::size_t max_num_points_in_pc)
      : ConcentricZoneModel(sensor_model, sensor_height, min_range, max_range),
          max_num_pts(max_num_points_in_pc)
  {
      // Initialize the GPU constants
      set_cnst_mem();

      std::transform(sensor_config_.num_laser_channels_per_zone_.begin(),
                      sensor_config_.num_laser_channels_per_zone_.end(),
                      std::back_inserter(sensor_config_.num_rings_for_each_zone_),
                     [](std::vector<int> &v) { return v.size();}
                    );

      mallocCuda();

#ifdef VIZ_PATCHES
      static constexpr uint8_t PALETTE[3][3] = {
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

  void mallocCuda()
  {
    // create 3d cuda memory for compacted patches as (ring, sector, points)
    max_num_sectors_ = *std::max_element(num_sectors_per_ring_.begin(),
                                                            num_sectors_per_ring_.end());

    // allocate memory for patches (ring, sector, points)
    cudaExtent extent_patches = make_cudaExtent(num_total_rings_*sizeof(PointT), max_num_sectors_, MAX_POINTS_PER_PATCH);
    CUDA_CHECK(cudaMalloc3D(&patches_d, extent_patches));
    patches_size = patches_d.pitch * patches_d.ysize * MAX_POINTS_PER_PATCH;


    // num of pts in each patch and pathces_d has the same layout as following.
    //     r0 | r1 | r2 | r3
    // s1|
    // s2|
    // s3|

    cudaExtent extent_num_pts_in_patch = make_cudaExtent(num_total_rings_ * sizeof(uint), max_num_sectors_, 1);
    CUDA_CHECK(cudaMalloc3D(&num_pts_in_patch_d, extent_num_pts_in_patch));
    num_pts_in_patch_size = num_pts_in_patch_d.pitch * num_pts_in_patch_d.ysize;

    CUDA_CHECK(cudaMalloc((void**)&cloud_in_d_, sizeof(PointT) * MAX_POINTS));

#ifdef VIZ_PATCHES
    CUDA_CHECK(cudaMallocHost((void**)&num_pts_in_patch_h, num_pts_in_patch_size));
    CUDA_CHECK(cudaMallocHost((void**)&patches_h, patches_size));
#endif // VIZ_PATCHES

    reset_buffers();
  }

  void reset_buffers(cudaStream_t stream=nullptr )
  {
    CUDA_CHECK(cudaMemsetAsync(patches_d.ptr, 0, patches_size, stream));
    CUDA_CHECK(cudaMemsetAsync(num_pts_in_patch_d.ptr, 0, num_pts_in_patch_size, stream));
    CUDA_CHECK(cudaMemsetAsync(cloud_in_d_, 0, sizeof(PointT) * MAX_POINTS, stream));
  }

  ConcentricZoneModelGPU() = default;

  ~ConcentricZoneModelGPU()
  {
    // Safely reset GPU device only if CUDA context is valid
    cudaFree(patches_d.ptr);  // only free device memory
    cudaFree(num_pts_in_patch_d.ptr);

    if (cloud_in_d_) {
      cudaFree(cloud_in_d_);
    }

#ifdef VIZ_PATCHES
    if (patches_h) {
      cudaFreeHost(patches_h);
    }

    if (num_pts_in_patch_h) {
      cudaFreeHost(num_pts_in_patch_h);
    }
#endif // VIZ_PATCHES
  }

  void set_cnst_mem();

  void to_CUDA( pcl::PointCloud<PointT>* pc, cudaStream_t stream=0)
  {
    CUDA_CHECK(cudaMemcpyAsync(cloud_in_d_, pc->points.data(), pc->points.size() * sizeof(PointT),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());
  }

#ifdef VIZ_PATCHES
  uint32_t cuda_patches_to_pcl( pcl::PointCloud<PointT>* pc, cudaStream_t stream=0)
  {
    CUDA_CHECK(cudaMemcpyAsync(patches_h, patches_d.ptr, patches_size,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(num_pts_in_patch_h, num_pts_in_patch_d.ptr, num_pts_in_patch_size,
                               cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);
    uint32_t num_patched_pts {0};
    for(int ring_idx=0; ring_idx<num_total_rings_; ring_idx++)
    {
      for(int sector_idx=0; sector_idx< max_num_sectors_; sector_idx++)
      {
        auto* rowPtr = reinterpret_cast<uint*>( num_pts_in_patch_h +
                                               sector_idx * num_pts_in_patch_d.pitch);
        uint num_pts = rowPtr[ring_idx];

        for(std::size_t pt_idx=0; pt_idx<num_pts; pt_idx++)
        {
          std::size_t row_offset = pt_idx * patches_d.ysize * patches_d.pitch +
                                  sector_idx * patches_d.pitch;
          auto* row = reinterpret_cast<PointT*>(patches_h + row_offset);
          PointT pt_loc = row[ring_idx];
          //encode ring,sector info as intensity to be colorized when visualized
          pt_loc.intensity = color_map[ring_idx * max_num_sectors_ + sector_idx];
          pc->points.push_back(pt_loc);
          num_patched_pts++;
        }
      }
    }
    return num_patched_pts;
  }
#endif // VIZ_PATCHES

  bool create_patches(pcl::PointCloud<PointT>* pc, cudaStream_t& stream)
  {
    to_CUDA(pc, stream);
    bool ret = launch_create_patches_kernel(pc->points.size(), stream);
    return ret;
  }

  bool launch_create_patches_kernel(int num_pc_pts, cudaStream_t& stream);

};

