//
// Created by orcun on 10.05.2025.
//

#ifndef PATCHWORK_CUDA_UTILS_CUH
#define PATCHWORK_CUDA_UTILS_CUH
#include <cuda_runtime_api.h>
#include <cuda.h>



#define MAX_POINTS 300000
#define MAX_POINTS_PER_PATCH 256
#define NUM_THREADS 256

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

int divup(int a, int b)
{
  /// Round up the division of a/b
  return (a + b - 1) / b;
}

#endif  // PATCHWORK_CUDA_UTILS_CUH
