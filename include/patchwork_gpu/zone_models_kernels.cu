//
// Created by orcun on 10.05.2025.
//

#include "zone_models_gpu.cuh"


inline __device__ float xy2sqr_r(const float &x, const float &y) { return x * x + y * y; }

inline __device__ float xy2theta(const float &x, const float &y) {  // 0 ~ 2 * PI
  auto atan_value = atan2f(y, x);       // EDITED!
  return atan_value > 0 ? atan_value : atan_value + 2 * M_PI;  // EDITED!
}

inline __device__ int get_ring_idx(const float &x, const float &y) {
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

inline __device__ int get_sector_idx(const float &x, const float &y, const int ring_idx) {
  float theta = xy2theta(x, y);
  int num_sectors = cnst_num_sectors_per_ring[ring_idx];
  float sector_size = 2.0 * M_PI / static_cast<float>(num_sectors);

  // min: for defensive programming
  return min(static_cast<int>(theta / sector_size), num_sectors - 1);
}

inline __device__ int2 get_ring_sector_idx(const float &x, const float &y)
{
  int ring_idx = get_ring_idx(x, y);
  if (ring_idx == INVALID_RING_IDX || ring_idx == OVERFLOWED_IDX) {
    return make_int2(ring_idx, ring_idx);
  }

  int sector_idx = get_sector_idx(x, y, ring_idx);
  return make_int2(ring_idx, sector_idx);
}

template<typename PointT>
__global__ void create_patches_kernel(const PointT *points, const int2* pt_patch_coord, int num_pts_in_cloud)
{
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_pts_in_cloud) {
    const PointT &pt = points[idx];
    pt_patch_coord[idx] = get_ring_sector_idx(pt.x(), pt.y());
  }
}
__global__ void pack_patches(const cudaPitchedPtr* patches,
                              const int2* pt_patch_coord,
                              const int2* pt_patch_coord_d,
                              const int* num_pts_in_patch)
{

}