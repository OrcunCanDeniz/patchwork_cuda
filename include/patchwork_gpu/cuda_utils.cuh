//
// Created by orcun on 10.05.2025.
//

#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>



#define MAX_POINTS 300000
#define MAX_POINTS_PER_PATCH 10000

extern __device__ __constant__ float cnst_sqr_boundary_ranges[256];
extern __device__ __constant__ std::size_t cnst_boundary_ranges_size;
extern __device__ __constant__ float cnst_sqr_max_range;
extern __device__ __constant__ int cnst_num_sectors_per_ring[256];
extern __device__ __constant__ std::size_t cnst_num_sectors_per_ring_size;

struct PointMeta {
    int ring_idx;  // ring index
    int sector_idx; // sector index
    uint lin_sec_idx; // patch index
    int iip{-1}; // intra-patch index
    float lbr{0.0f}; // low point representative (LPR) for the patch
    bool ground{false}; // whether this point is a seed point
};

__device__ size_t resolve_lin_sec_idx(int ring_idx, int sector_idx)
{
  // Calculate linear sector index based on ring and sector indices
  uint lin_sector_idx{0};
  for (int i=0; i<ring_idx; ++i) {
    lin_sector_idx += cnst_num_sectors_per_ring[i];
  }
  return lin_sector_idx + sector_idx;
}

__device__ uint2 ring_sec_idx_from_lin_idx(uint lin_sec_idx)
{
  // Get ring and sector indices from linear sector index
  uint ring_idx{0};
  while(lin_sec_idx > cnst_num_sectors_per_ring[ring_idx]-1){
    lin_sec_idx -= cnst_num_sectors_per_ring[ring_idx];
    ring_idx++;
  }
  return make_uint2(ring_idx, lin_sec_idx);
}

__device__ __host__ inline PointMeta make_point_meta(int ring_idx, int sector_idx,
                                                     uint lin_sec_idx, int iip)
{
    PointMeta meta;
    meta.ring_idx = ring_idx;
    meta.sector_idx = sector_idx;
    meta.lin_sec_idx = lin_sec_idx;
    meta.iip = iip;
    return meta;
}

//define cuda error checker
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

inline __host__ __device__ int divup(int a, int b)
{
  // Round up the division of a/b
  return (a + b - 1) / b;
}

#endif  // CUDA_UTILS_CUH
