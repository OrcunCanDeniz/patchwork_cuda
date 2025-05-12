//
// Created by orcun on 10.05.2025.
//

#ifndef PATCHWORK_ZONE_MODELS_GPU_CUH
#define PATCHWORK_ZONE_MODELS_GPU_CUH
#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "patchwork/zone_models.hpp"

#include "cuda_utils.cuh"
#include "pcl/point_cloud.h"
__constant__ float cnst_sqr_boundary_ranges[256];
__constant__ std::size_t cnst_boundary_ranges_size;
__constant__ float cnst_sqr_max_range;
__constant__ int cnst_num_sectors_per_ring[256];
__constant__ std::size_t cnst_num_sectors_per_ring_size;

template<typename PointT>
class ConcentricZoneModelGPU: public ConcentricZoneModel
{
  std::size_t max_num_pts{0};
  cudaPitchedPtr* patches_d{nullptr};
  std::size_t patches_size{0};
  cudaPitchedPtr* num_pts_in_patch_d{nullptr};
  std::size_t num_pts_in_patch_size{0};
  PointT* cloud_in_d_{nullptr};

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
  }

  void mallocCuda()
  {
    // create 3d cuda memory for compacted patches as (ring, sector, points)
    std::size_t max_sector_num_in_ring = *std::max_element(num_sectors_per_ring_.begin(),
                                                            num_sectors_per_ring_.end());

    // allocate memory for patches (ring, sector, points)
    cudaExtent extent_patches = make_cudaExtent(num_total_rings_*sizeof(PointT),
                                                max_sector_num_in_ring, MAX_POINTS_PER_PATCH);
    CUDA_CHECK(cudaMalloc3D(patches_d, extent_patches));
    patches_size = patches_d->pitch * max_sector_num_in_ring * MAX_POINTS_PER_PATCH;

    // allocate memory for num of pts in each patch
    cudaExtent extent_num_pts_in_patch = make_cudaExtent(num_total_rings_ * sizeof(uint16_t),
                                                    max_sector_num_in_ring, 1);
    CUDA_CHECK(cudaMalloc3D(num_pts_in_patch_d, extent_num_pts_in_patch));
    num_pts_in_patch_size = num_pts_in_patch_d->pitch * max_sector_num_in_ring;

    CUDA_CHECK(cudaMalloc((void**)&cloud_in_d_, sizeof(PointT) * MAX_POINTS));

    reset_buffers();
  }

  void reset_buffers(cudaStream_t stream=nullptr )
  {
    CUDA_CHECK(cudaMemsetAsync(patches_d->ptr, 0, patches_size, stream));
    CUDA_CHECK(cudaMemsetAsync(num_pts_in_patch_d, 0, num_pts_in_patch_size, stream));
    CUDA_CHECK(cudaMemsetAsync(cloud_in_d_, 0, sizeof(PointT) * MAX_POINTS, stream));
  }

  ConcentricZoneModelGPU() = default;

  ~ConcentricZoneModelGPU()
  {
    if (patches_d) {
      CUDA_CHECK(cudaFree(patches_d->ptr));
      free(patches_d);
    }
    if (num_pts_in_patch_d) {
      CUDA_CHECK(cudaFree(num_pts_in_patch_d->ptr));
      free(num_pts_in_patch_d);
    }
  }

  void set_cnst_mem()
  {
    CUDA_CHECK(cudaMemcpyToSymbol(cnst_sqr_boundary_ranges, sqr_boundary_ranges_.data(),
                                  sizeof(float) * sqr_boundary_ranges_.size()));
    auto tmp = sqr_boundary_ranges_.size();
    CUDA_CHECK(cudaMemcpyToSymbol(&cnst_boundary_ranges_size, &tmp, sizeof(std::size_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(&cnst_sqr_max_range, &sqr_max_range_, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(&cnst_num_sectors_per_ring, num_sectors_per_ring_.data(),
                                  sizeof(int) * num_sectors_per_ring_.size()));
  }

  void toCUDA( pcl::PointCloud<PointT> pc, cudaStream_t stream=0)
  {
    CUDA_CHECK(cudaMemcpyAsync(cloud_in_d_, pc.points.data(), pc.points.size(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());
  }

  bool launch_create_patches_kernel(PointT* in_points, int num_pc_pts, cudaStream_t& stream);

  //TODO: write function to copy patches to host maybe thrust::copy_if
};

#endif  // PATCHWORK_ZONE_MODELS_GPU_CUH
