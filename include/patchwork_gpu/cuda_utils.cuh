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
#define NUM_THREADS 512

extern __device__ __constant__ float cnst_sqr_boundary_ranges[256];
extern __device__ __constant__ std::size_t cnst_boundary_ranges_size;
extern __device__ __constant__ float cnst_sqr_max_range;
extern __device__ __constant__ int cnst_num_sectors_per_ring[256];

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

inline int divup(int a, int b)
{
  /// Round up the division of a/b
  return (a + b - 1) / b;
}

#endif  // CUDA_UTILS_CUH
