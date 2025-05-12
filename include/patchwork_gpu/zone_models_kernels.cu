//
// Created by orcun on 10.05.2025.
//

#include "zone_models_gpu.cuh"

// __device__ functions are inlined by default

__device__ float xy2sqr_r(const float &x, const float &y) { return x * x + y * y; }

__device__ float xy2theta(const float &x, const float &y) {  // 0 ~ 2 * PI
  auto atan_value = atan2f(y, x);       // EDITED!
  return atan_value > 0 ? atan_value : atan_value + 2 * M_PI;  // EDITED!
}

__device__ int get_ring_idx(const float &x, const float &y) {
  float sqr_r = xy2sqr_r(x, y);
  // Exception for UAVs such as NTU VIRAL dataset
  if (sqr_r < cnst_sqr_boundary_ranges[0]) {
    return INVALID_RING_IDX;
  }
  if (sqr_r > cnst_sqr_max_range) {
    return OVERFLOWED_IDX;
  }

  for (int i = 1; i < cnst_boundary_ranges_size; ++i) {
    if (sqr_r <= cnst_sqr_boundary_ranges[i]) {
      return i-1;
    }
  }
}

__device__ int get_sector_idx(const float &x, const float &y, const int ring_idx) {
  float theta = xy2theta(x, y);
  int num_sectors = cnst_num_sectors_per_ring[ring_idx];
  float sector_size = 2.0 * M_PI / static_cast<float>(num_sectors);

  // min: for defensive programming
  return min(static_cast<int>(theta / sector_size), num_sectors - 1);
}

__device__ int2 get_ring_sector_idx(const float &x, const float &y)
{
  int ring_idx = get_ring_idx(x, y);
  if (ring_idx == INVALID_RING_IDX || ring_idx == OVERFLOWED_IDX) {
    return make_int2(ring_idx, ring_idx);
  }

  int sector_idx = get_sector_idx(x, y, ring_idx);
  return make_int2(ring_idx, sector_idx);
}

template<typename PointT>
__global__ void create_patches_kernel(const PointT *points, const cudaPitchedPtr* patches,
                                      unsigned int* num_pts_in_patch,
                                      int num_pts_in_cloud, std::size_t ring_step)
{
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pts_in_cloud) return;

  const PointT &pt = points[idx];
  int2 ring_sector_indices = get_ring_sector_idx(pt.x(), pt.y());
  const std::size_t patch_numel_offset = ring_step*ring_sector_indices.x + ring_sector_indices.y;
  const int cur_num_pts_in_patch = atomicAdd((num_pts_in_patch + patch_numel_offset), 1);

  if (cur_num_pts_in_patch >= MAX_POINTS_PER_PATCH) return;
  // TODO maybe add a marking mechanism to see if any patches are overflowed

  const std::size_t row_offset = cur_num_pts_in_patch * patches->ysize * patches->pitch +
                                   ring_sector_indices.y * patches->pitch ;
  const char* row = (const char*)patches->ptr + row_offset;
  PointT* pt_loc = (PointT *)row + ring_sector_indices.x;
  *pt_loc = pt;
}

template<typename PointT>
bool ConcentricZoneModelGPU<PointT>::launch_create_patches_kernel(PointT* in_points,
                                                                  int num_pc_pts, cudaStream_t& stream)
{
  if (num_pc_pts > max_num_pts) {
    throw std::runtime_error("Number of points in the point cloud exceeds the maximum limit.");
  }

  // Launch the kernel
  dim3 threads(NUM_THREADS);
  dim3 blocks(divup(num_pc_pts, NUM_THREADS));

  create_patches_kernel<<<blocks, threads, 0, stream>>>(in_points, patches_d,
                                                        num_pc_pts);
  cudaStreamSynchronize(stream);
  CUDA_CHECK(cudaGetLastError());
}

