//
// Created by orcun on 10.05.2025.
//

#include "patchwork_gpu/zone_models_gpu.cuh"
// __device__ functions are inlined by default

__device__ __constant__ float cnst_sqr_boundary_ranges[256];
__device__ __constant__ std::size_t cnst_boundary_ranges_size;
__device__ __constant__ float cnst_sqr_max_range;
__device__ __constant__ int cnst_num_sectors_per_ring[256];


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
__global__ void create_patches_kernel( PointT *points, const cudaPitchedPtr patches,
                                      const cudaPitchedPtr num_pts_in_patch, float z_thresh,
                                      int num_pts_in_cloud)
{
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pts_in_cloud) return;

  const PointT &pt = points[idx];
  if (pt.z < z_thresh) return;

  int2 ring_sector_indices = get_ring_sector_idx(pt.x, pt.y);

  if (ring_sector_indices.x < 0 ) return;

  std::size_t offsetBytes = num_pts_in_patch.pitch * ring_sector_indices.y + ring_sector_indices.x * sizeof(uint);

  uint* patch_numel_ptr = reinterpret_cast<uint*>(static_cast<char*>(num_pts_in_patch.ptr) + offsetBytes);
  const uint cur_num_pts_in_patch = atomicAdd(patch_numel_ptr, 1);
// this increments the patch_numel_ptr irrelevant of its previous value, be careful when reading it.

  if (cur_num_pts_in_patch >= MAX_POINTS_PER_PATCH) return;

  // calculate where to place the point in the patch
  const std::size_t row_offset = cur_num_pts_in_patch * patches.ysize * patches.pitch +
                                   ring_sector_indices.y * patches.pitch ;
  const char* row = (const char*)patches.ptr + row_offset;
  // using float4 to store point for internal representation, more compact and mem-=aligned
  float4* pt_loc = (float4 *)row + ring_sector_indices.x;
  *pt_loc = make_float4(pt.x, pt.y, pt.z, pt.intensity); // store the point into patch
}

template<typename PointT>
bool ConcentricZoneModelGPU<PointT>::launch_create_patches_kernel(PointT* cloud_in_d,
                                                                  int num_pc_pts,
                                                                  cudaPitchedPtr& patches_d,
                                                                  cudaPitchedPtr& num_pts_in_patch_d,
                                                                  cudaStream_t& stream)
{
  if (num_pc_pts > max_num_pts) {
    throw std::runtime_error("Number of points in the point cloud exceeds the maximum limit.");
  }

  // TODO : definition of z_thresh might change. come back here later
  float z_thresh = -sensor_height_ - 2.0; // threshold for z coordinate

  // Launch the kernel
  dim3 threads(NUM_THREADS);
  dim3 blocks(divup(num_pc_pts, NUM_THREADS));

  std::cout<< "Num points in OG cloud: " << num_pc_pts << std::endl;

  create_patches_kernel<<<blocks, threads, 0, stream>>>(cloud_in_d, patches_d,
                                                        num_pts_in_patch_d, z_thresh,
                                                        num_pc_pts);

  auto err = cudaGetLastError();
  CUDA_CHECK(err);

  return err == cudaSuccess;
}

template<typename PointT>
void ConcentricZoneModelGPU<PointT>::set_cnst_mem()
{
  // we declared __constant__ symbols as extern in cuh but didnt  defined them.
  // also only compiled this file, so symbols wouldnt be compiled if they didn't reside in cpp/cu file
  // as a result this function or at least one reference of each symbol must have been in cu/cpp file
  // that's why just this func is here cu file.
  CUDA_CHECK(cudaMemcpyToSymbol(cnst_sqr_boundary_ranges, sqr_boundary_ranges_.data(),
                                sizeof(float) * sqr_boundary_ranges_.size()));
  auto tmp = sqr_boundary_ranges_.size();
  CUDA_CHECK(cudaMemcpyToSymbol(cnst_boundary_ranges_size, &tmp, sizeof(std::size_t)));
  CUDA_CHECK(cudaMemcpyToSymbol(cnst_sqr_max_range, &sqr_max_range_, sizeof(float)));
  CUDA_CHECK(cudaMemcpyToSymbol(cnst_num_sectors_per_ring, num_sectors_per_ring_.data(),
                                sizeof(int) * num_sectors_per_ring_.size()));
}

template class ConcentricZoneModelGPU<pcl::PointXYZI>;
template class ConcentricZoneModelGPU<PointXYZILID>;

