//
// Created by orcun on 25.05.2025.
//

#include "patchwork_gpu/patchwork_gpu.cuh"
#define NUM_PATCHES_PER_BLOCK 4
#define NUM_PTS_PER_THREAD 32

__device__ __constant__ int cnst_num_sectors_per_ring[256];
__device__ __constant__ std::size_t cnst_num_sectors_per_ring_size;



__global__ void per_patch_lbr_compute(cudaPitchedPtr patches, const uint2 ring_sector_ids,
                                      const float z_thresh, const int num_pts_in_patch,
                                      const bool close_zone, const cudaPitchedPtr lbr)
{
  __shared__ float z_mean;
  __shared__ uint valid_pts_in_patch;
  const uint thread_in_patch = threadIdx.x;

  for(uint pt_idx=thread_in_patch ; pt_idx < num_pts_in_patch; pt_idx += NUM_PTS_PER_THREAD)
  {
    // get the point in patch from ring_sector_ids
    const std::size_t row_offset = patches.pitch * patches.ysize * pt_idx +
                              patches.pitch * ring_sector_ids.y;
    const auto* row_ptr = reinterpret_cast<float4*>(static_cast<char*>(patches.ptr) + row_offset);
    float4 pt = row_ptr[ring_sector_ids.x];
    if (close_zone) // TODO shouldnt cause warp divergence but recheck and ensure
    {
      if (pt.z > z_thresh) {
        atomicAdd(&z_mean, pt.z);
        atomicAdd(&valid_pts_in_patch, 1);
      }
    } else {
      atomicAdd(&z_mean, pt.z);
      atomicAdd(&valid_pts_in_patch, 1);
    }
  }
  __syncthreads(); // sync all threads in patch
  if (thread_in_patch == 0) {
      // compute the mean height of the patch
      const bool empty_patch = (valid_pts_in_patch == 0);
      const float tmp_norm = empty_patch > 0 ? static_cast<float>(valid_pts_in_patch) : 1.0f; // avoid division by zero
      z_mean = z_mean / tmp_norm;

      auto lbr_row = reinterpret_cast<float*>( static_cast<char*>(lbr.ptr) +
                                                 lbr.pitch * ring_sector_ids.y );
      lbr_row[ring_sector_ids.x] = empty_patch ? 0.0f : z_mean;
  }
}

__global__ void per_patch_manager_kernel(const cudaPitchedPtr patches_ptr,
                                          const cudaPitchedPtr num_pts_in_patch_ptr,
                                          const float z_thres,
                                          const int max_ring_idx_in_first_zone,
                                          const uint min_pts_thres,
                                         cudaPitchedPtr lbr_d)
{
//  const int* num_sectors_per_ring, set as consant
  const uint ring_idx = threadIdx.x;
  const uint sector_idx = threadIdx.y;
  const uint2 ring_sector_ids = make_uint2(ring_idx, sector_idx);

  // some sectors actually does not exist, they're just here to keep a 3d data structure
  const bool dummy_sector = sector_idx >= cnst_num_sectors_per_ring[ring_idx];

  auto* num_ptr = reinterpret_cast<int*>(
                          static_cast<char*>(num_pts_in_patch_ptr.ptr) +
                                num_pts_in_patch_ptr.pitch * sector_idx
                          ) + ring_idx;
  if (dummy_sector)
  {
    *num_ptr = 0; // no points in this sector
    return;
  }
  const int num_pts_in_patch = *num_ptr;
  const bool is_close = (ring_idx <= max_ring_idx_in_first_zone);
  const bool few_points = (num_pts_in_patch < min_pts_thres);

  if(few_points) return;
  // TODO do not forget about few_points case
  dim3 threads( divup(num_pts_in_patch, NUM_PTS_PER_THREAD) );

  per_patch_lbr_compute<<<threads, 1>>>(patches_ptr, ring_sector_ids, z_thres,
                                        num_pts_in_patch, is_close, lbr_d);

}

template <typename PointT>
void PatchWorkGPU<PointT>::launch_seed_extract_kernel(cudaStream_t& stream)
{
  static double lowest_h_margin_in_close_zone =
      (sensor_height_ == 0.0) ? -0.1 : adaptive_seed_selection_margin_ * sensor_height_;

  // for patches in first zone, we only consider the points that are above the sensor height
  // for patches in other zones, all points are used to calculate mean height in patch
  // variable num of threads per patch may be useful.
  dim3 threads(zone_model_->num_total_rings_, zone_model_->max_num_sectors_);
  per_patch_manager_kernel<<<threads, 1, 0, stream>>>(
      patches_d, num_pts_in_patch_d, lowest_h_margin_in_close_zone,
      zone_model_->max_ring_idx_in_first_zone_, num_min_pts_);

}