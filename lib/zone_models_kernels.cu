//
// Created by orcun on 10.05.2025.
//

#include "patchwork_gpu/zone_models_gpu.cuh"
#include <cub/cub.cuh>

// __device__ functions are inlined by default

__device__ __constant__ float cnst_sqr_boundary_ranges[256];
__device__ __constant__ std::size_t cnst_boundary_ranges_size;
__device__ __constant__ float cnst_sqr_max_range;
__device__ __constant__ int cnst_num_sectors_per_ring[256];
__device__ __constant__ std::size_t cnst_num_sectors_per_ring_size;


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
__global__ void count_patches_kernel( PointT *points,
                                      uint* num_pts_in_patch,
                                      PointMeta* metas,
                                     float z_thresh,
                                      int num_pts_in_cloud)
{
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pts_in_cloud) return;

  const PointT &pt = points[idx];
  if (pt.z < z_thresh) return;

  int2 ring_sector_indices = get_ring_sector_idx(pt.x, pt.y);

  const size_t lin_sector_idx = resolve_lin_sec_idx(ring_sector_indices.x, ring_sector_indices.y);
  uint* patch_numel_ptr = num_pts_in_patch + lin_sector_idx;
  int iip = -1; // intra-patch index

  if (ring_sector_indices.x >= 0 )
  {
   iip = atomicAdd(patch_numel_ptr, 1); // save this as idx in patch
  }

  metas[idx] = make_point_meta( ring_sector_indices.x,
                               ring_sector_indices.y,
                               lin_sector_idx,
                               iip);
}

template<typename PointT>
__global__ void move_points_to_patch_kernel(PointT* points,
                                            const PointMeta* metas_d,
                                            const uint* offsets_d,
                                            float4* patches_d, float z_thresh,
                                            uint num_pc_points) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pc_points) return;

  const PointT &pt = points[idx];
  if (pt.z < z_thresh) return;
  PointMeta meta = metas_d[idx];
  if (meta.iip == -1) return;
  float4 pt4 = make_float4(pt.x, pt.y, pt.z, pt.intensity);
  patches_d[offsets_d[meta.lin_sec_idx] + meta.iip] = pt4;
}

template<typename PointT>
bool ConcentricZoneModelGPU<PointT>::create_patches_gpu(PointT* cloud_in_d, int num_pc_pts,
                                                          uint* num_pts_in_patch_d,
                                                          PointMeta* metas_d,
                                                          uint* offsets_d,
                                                          uint num_total_sectors,
                                                          float4* patches_d,
                                                          cudaStream_t& stream)
{
  if (num_pc_pts > max_num_pts) {
    throw std::runtime_error("Number of points in the point cloud exceeds the maximum limit.");
  }

  // TODO : definition of z_thresh might change. come back here later
  float z_thresh = -sensor_height_ - 2.0; // threshold for z coordinate

  static const uint num_threads = 512;
  dim3 threads(num_threads);
  dim3 blocks(divup(num_pc_pts, num_threads));

  std::cout<< "Num points in OG cloud: " << num_pc_pts << std::endl;

  if (cub_dev_scan_sum_tmp_ != nullptr) {
    // this scratch memory must be replaced every time since num points is not consistent
    cudaFree(cub_dev_scan_sum_tmp_);
    cub_dev_scan_sum_tmp_ = nullptr;
  }

  count_patches_kernel<<<blocks, threads, 0, stream>>>(cloud_in_d,
                                                        num_pts_in_patch_d,
                                                        metas_d,
                                                        z_thresh,
                                                        num_pc_pts);
  cudaStreamSynchronize(stream);
  CUDA_CHECK(cudaGetLastError());

  // query the temporary storage size for the exclusive sum
  CUDA_CHECK( cub::DeviceScan::ExclusiveSum(
                      /* d_temp_storage */ nullptr,
                      /* temp_storage_bytes */ cub_dev_scan_sum_tmp_bytes,
                      /* d_in */ num_pts_in_patch_d,
                      /* d_out */ offsets_d,
                      /* num_items */ num_total_sectors,
                      /* stream */ stream)
  );

  cudaStreamSynchronize(stream);
  CUDA_CHECK(cudaGetLastError());

  // allocate temporary storage for exclusive sum
  CUDA_CHECK(cudaMalloc(&cub_dev_scan_sum_tmp_, cub_dev_scan_sum_tmp_bytes));

  // 3) run the scan
  CUDA_CHECK( cub::DeviceScan::ExclusiveSum(
                  /* d_temp_storage */    cub_dev_scan_sum_tmp_,
                  /* temp_storage_bytes */ cub_dev_scan_sum_tmp_bytes,
                  /* d_in */              num_pts_in_patch_d,
                  /* d_out */             offsets_d,
                  /* num_items */         num_total_sectors,
                  /* stream */            stream
              ));
  cudaStreamSynchronize(stream);
  CUDA_CHECK(cudaGetLastError());

  dim3 move_threads(num_threads);
  dim3 move_blocks(divup(num_pc_pts, num_threads));

  move_points_to_patch_kernel<<<move_blocks, move_threads,0, stream>>>(cloud_in_d, metas_d,
                                                                       offsets_d, patches_d,
                                                                        z_thresh, num_pc_pts);

  cudaStreamSynchronize(stream);
  CUDA_CHECK(cudaGetLastError());

  return true;
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
  auto tmp2 = num_sectors_per_ring_.size();
  CUDA_CHECK(cudaMemcpyToSymbol(cnst_num_sectors_per_ring_size, &tmp2,sizeof(std::size_t)));
}

template class ConcentricZoneModelGPU<pcl::PointXYZI>;
template class ConcentricZoneModelGPU<PointXYZILID>;

