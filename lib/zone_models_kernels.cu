//
// Created by orcun on 10.05.2025.
//
#include <cub/cub.cuh>
#include "patchwork_gpu/zone_models_gpu.cuh"

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
                                      PointMeta* in_metas,
                                      float z_thresh,
                                      int num_pts_in_cloud)
{
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pts_in_cloud) return;

  const PointT &pt = points[idx];
  int2 ring_sector_indices = get_ring_sector_idx(pt.x, pt.y);

  const size_t lin_sector_idx = resolve_lin_sec_idx(ring_sector_indices.x, ring_sector_indices.y);
  uint* patch_numel_ptr = num_pts_in_patch + lin_sector_idx;
  int iip = -1; // intra-patch index

  if (ring_sector_indices.x >= 0 && pt.z > z_thresh)
  {
   iip = atomicAdd(patch_numel_ptr, 1); // save this as idx in patch
  }

  in_metas[idx] = make_point_meta( ring_sector_indices.x,
                               ring_sector_indices.y,
                               lin_sector_idx,
                               iip);
}

template<typename PointT>
__global__ void move_points_to_patch_kernel(PointT* points,
                                            const PointMeta* in_metas_d,
                                            PointMeta* metas_d,
                                            float* z_keys,
                                            const uint* offsets_d,
                                            PointT* patches_d, float z_thresh,
                                            uint num_pc_points) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pc_points) return;

  const PointT &pt = points[idx];
//  if (pt.z < z_thresh) return; //  this is obsolete. meta.iip already validates points
  PointMeta meta = in_metas_d[idx];
  if (meta.iip == -1) return;
  const auto pt_offset = offsets_d[meta.lin_sec_idx] + meta.iip;
  patches_d[pt_offset] = pt;
  metas_d[pt_offset] = meta;
  z_keys[pt_offset] = pt.z;
}

template<typename PointT>
__global__ void apply_permutation_kernel(const PointT* in_patches_d,
                                          const PointMeta* in_metas_d,
                                          PointT* sorted_patches_d,
                                          PointMeta* sorted_metas_d,
                                          const uint* permutation,
                                          uint num_pc_points) {
  const uint dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (dst_idx >= num_pc_points) return;
  const uint src_idx = permutation[dst_idx];

  const PointMeta& meta = in_metas_d[src_idx];
  if (meta.iip == -1) return;
  const PointT& pt = in_patches_d[src_idx];
  sorted_patches_d[dst_idx] = pt;
  sorted_metas_d[dst_idx] = meta;
}


template<typename PointT>
bool ConcentricZoneModelGPU<PointT>::create_patches_gpu(PointT* cloud_in_d, int num_pc_pts,
                                                          uint* num_pts_in_patch_d,
                                                          PointMeta* in_metas_d,
                                                          PointMeta* metas_d,
                                                          uint* offsets_d,
                                                          uint num_total_sectors,
                                                          PointT* patches_d,
                                                          uint& num_patched_pts_h,
                                                          float* sorted_z_d,
                                                          cudaStream_t& stream)
{
  if (num_pc_pts > max_num_pts) {
    throw std::runtime_error("Number of points in the point cloud exceeds the maximum limit.");
  }

  float z_thresh = -sensor_height_ - 2.0; // threshold for z coordinate

  static const uint num_threads = 512;
  dim3 threads(num_threads);
  dim3 blocks(divup(num_pc_pts, num_threads));

  if (cub_dev_scan_sum_tmp_ != nullptr) {
    // this scratch memory must be replaced every time since num points is not consistent
    cudaFree(cub_dev_scan_sum_tmp_);
    cub_dev_scan_sum_tmp_ = nullptr;
  }
  // compute the num of points in each patch
  count_patches_kernel<<<blocks, threads, 0, stream>>>(cloud_in_d,
                                                        num_pts_in_patch_d,
                                                        in_metas_d,
                                                        z_thresh,
                                                        num_pc_pts);

  // compute patch offsets
  // query the temporary storage size for the exclusive sum
  static size_t sum_query_bytes{0};
  CUDA_CHECK( cub::DeviceScan::InclusiveSum(
                      /* d_temp_storage */ nullptr,
                      /* temp_storage_bytes */ sum_query_bytes,
                      /* d_in */ num_pts_in_patch_d,
                      /* d_out */ offsets_d+1,
                      /* num_items */ num_total_sectors,
                      /* stream */ stream)
  );
  // to have both inclusive and exclusive sum, we have num_segments+1 buffer. starting to write 1st
//  element (instead of 0th) we keep the first patch offset as zero, but also keep the correct buffer end
//  at num_segments+1th element. End offset is specifically needed for segmented sort
  if (cub_dev_scan_sum_tmp_ == nullptr
      || sum_query_bytes > cub_dev_scan_sum_tmp_bytes) {

    if (cub_dev_scan_sum_tmp_) {
      CUDA_CHECK(cudaFreeAsync(cub_dev_scan_sum_tmp_, stream));
    }
    // Allocate exactly what CUB needs
    cub_dev_scan_sum_tmp_bytes = sum_query_bytes;
    CUDA_CHECK(cudaMallocAsync(&cub_dev_scan_sum_tmp_, cub_dev_scan_sum_tmp_bytes, stream));
  }

  CUDA_CHECK( cub::DeviceScan::InclusiveSum(
                  /* d_temp_storage */    cub_dev_scan_sum_tmp_,
                  /* temp_storage_bytes */ cub_dev_scan_sum_tmp_bytes,
                  /* d_in */              num_pts_in_patch_d,
                  /* d_out */             offsets_d+1,
                  /* num_items */         num_total_sectors,
                  /* stream */            stream
              ));
  cudaStreamSynchronize(stream); // end compute offsets
  CUDA_CHECK(cudaGetLastError());

  cudaMemcpyAsync(num_pts_per_patch_h.data(), num_pts_in_patch_d,
                  sizeof(uint) * num_total_sectors, cudaMemcpyDeviceToHost, czm_stream_);

  dim3 move_threads(num_threads);
  dim3 move_blocks(divup(num_pc_pts, num_threads));
  // move points from input cloud to patches buffer
  move_points_to_patch_kernel<<<move_blocks, move_threads,0, stream>>>(cloud_in_d,
                                                                        in_metas_d,
                                                                        metas_interm,
                                                                        z_keys_d_,
                                                                        offsets_d, unsorted_patches_d_,
                                                                        z_thresh, num_pc_pts);
  cudaStreamSynchronize(czm_stream_);
  num_patched_pts_h = std::accumulate(num_pts_per_patch_h.begin(), num_pts_per_patch_h.end(), 0u);

  // sort workspace size just depends on num_total_sectors,
  //  thus the workspace can be reused through the lifetime of the program
  static size_t sort_query_bytes = 0;
  cub::DeviceSegmentedSort::SortPairs(
                                      nullptr, sort_query_bytes,
                                      z_keys_d_, sorted_z_d,
                                      raw_perm_idx, sorted_perm_idx,
                                      num_patched_pts_h, num_total_sectors,
                                      offsets_d, offsets_d + 1, stream);

  if (cub_sort_tmp_d == nullptr || sort_query_bytes > cub_sort_tmp_bytes)
  {
    // if prev scratch pad allocation is not enough, free and realloc
    if (cub_sort_tmp_d) cudaFreeAsync(cub_sort_tmp_d, stream);
    cub_sort_tmp_bytes = sort_query_bytes;
    // Allocate temporary storage
    cudaMallocAsync(&cub_sort_tmp_d, cub_sort_tmp_bytes, stream);
  }
  // sort pts within patches by z. only get idx mapping
  cub::DeviceSegmentedSort::SortPairs(
      cub_sort_tmp_d, cub_sort_tmp_bytes,
      z_keys_d_, sorted_z_d, raw_perm_idx, sorted_perm_idx,
      num_patched_pts_h, num_total_sectors, offsets_d, offsets_d + 1, stream);

  dim3 perm_threads(num_threads);
  dim3 perm_blocks(divup(num_patched_pts_h, num_threads));
  apply_permutation_kernel<<<perm_blocks, perm_threads, 0, stream>>>(unsorted_patches_d_, metas_interm,
                                                                     patches_d, metas_d,
                                                                     sorted_perm_idx, num_patched_pts_h
                                                                     );

  return true;
}

__global__ void set_lin_inc_mem(uint* dst, const uint numel)
{
  const uint tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid>=numel) return;
  dst[tid] = tid;
}

template<typename PointT>
void ConcentricZoneModelGPU<PointT>::set_permute_idx()
{
  dim3 threads(512);
  dim3 blocks(divup(max_num_pts, 512));

  set_lin_inc_mem<<<blocks, threads, 0, czm_stream_>>>(raw_perm_idx, max_num_pts);
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

template class ConcentricZoneModelGPU<PointXYZILID>;

