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

#define INVALID_RING_IDX -1
#define OVERFLOWED_IDX -2

__constant__ float cnst_sqr_boundary_ranges[256];
__constant__ std::size_t cnst_boundary_ranges_size;
__constant__ float cnst_sqr_max_range;
__constant__ int cnst_num_sectors_per_ring[256];
__constant__ std::size_t cnst_num_sectors_per_ring_size;

template<typename PointT>
class ConcentricZoneModelGPU: public ConcentricZoneModel
{
  std::size_t max_num_pts{0};
  int2 *pt_patch_coord_d{nullptr};
  cudaPitchedPtr* patches_d{nullptr};

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
    CUDA_CHECK(cudaMalloc((void**)&pt_patch_coord_d, sizeof(int2) * max_num_pts));

    // create 3d cuda memory for compacted patches as (ring, sector, points)
    std::size_t total_ring_num = std::accumulate(sensor_config_.num_rings_for_each_zone_.begin(),
                                                  sensor_config_.num_rings_for_each_zone_.end(), 0);
    std::size_t total_sector_num = std::accumulate(sensor_config_.num_sectors_for_each_zone_.begin(),
                                                   sensor_config_.num_sectors_for_each_zone_.end(), 0);
    cudaExtent extent = make_cudaExtent(total_ring_num*sizeof(PointT), total_sector_num, MAX_POINTS_PER_PATCH);
    CUDA_CHECK(cudaMalloc3D(patches_d, extent));
  }

  ConcentricZoneModelGPU() = default;

  ~ConcentricZoneModelGPU()
  {
    if (pt_patch_coord_d != nullptr) {
      cudaFree(pt_patch_coord_d);
      pt_patch_coord_d = nullptr;
    }
  }

  void set_cnst_mem()
  {
    CUDA_CHECK(cudaMemcpyToSymbol(cnst_sqr_boundary_ranges, sqr_boundary_ranges_.data(),
                                  sizeof(float) * sqr_boundary_ranges_.size()));
    auto tmp = sqr_boundary_ranges_.size();
    CUDA_CHECK(cudaMemcpyToSymbol(cnst_boundary_ranges_size, &tmp, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(cnst_sqr_max_range, &sqr_max_range_, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(cnst_num_sectors_per_ring, num_sectors_per_ring_.data(),
                                  sizeof(int) * num_sectors_per_ring_.size()));
    auto tmp2 = num_sectors_per_ring_.size();
    CUDA_CHECK(cudaMemcpyToSymbol(cnst_num_sectors_per_ring_size, &tmp2, sizeof(int)));
  }

  template<typename PointT>
  bool launch_create_patches_kernel(PointT* in_points, int num_pc_pts, cudaStream_t& stream)
  {
    if (num_pc_pts > max_num_pts) {
      throw std::runtime_error("Number of points in the point cloud exceeds the maximum limit.");
    }

    // Launch the kernel
    dim3 threads(NUM_THREADS);
    dim3 blocks(divup(num_pc_pts, NUM_THREADS));

    create_patches_kernel<<<blocks, threads, 0, stream>>>(in_points, pt_patch_coord_d, num_pc_pts);
    cudaStreamSynchronize(stream);
    CUDA_CHECK(cudaGetLastError());
  }
};

#endif  // PATCHWORK_ZONE_MODELS_GPU_CUH
