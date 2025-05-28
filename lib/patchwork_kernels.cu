//
// Created by orcun on 25.05.2025.
//

#include "patchwork_gpu/patchwork_gpu.cuh"
#define NUM_THREADS_PER_PATCH 128

__device__ __constant__ int cnst_num_sectors_per_ring[256];
__device__ __constant__ std::size_t cnst_num_sectors_per_ring_size;
__device__ __constant__ float cnst_lbr_margin;

// Single kernel version: one block per patch with parallel reduction in shared memory
__global__ void lbr_seed_kernel(
    const float4* patches,
    const uint* num_pts,
    const uint* offsets,
    const double close_zone_z_thresh,
    const int max_ring_first,
    const uint min_pts_thres,
    PointMeta* metas)
{
  const int patch_idx = blockIdx.x;
  const uint n = num_pts[patch_idx];

  if (n == 0) return;
  const bool all_ground = n < min_pts_thres;

  const size_t offset = offsets[patch_idx];
  const bool close_zone = (patch_idx < max_ring_first);

  extern __shared__ float shared_mem[];
  // split shared mem to 2 chunks
  float* sum_buf = shared_mem;
  uint* cnt_buf = (unsigned int*)&sum_buf[blockDim.x];

  const int tid = threadIdx.x;
  float local_sum = 0.0f;
  uint local_cnt = 0;

  for(uint i = tid; i < n; i += blockDim.x){
    float z = patches[offset + i].z;
    if(!close_zone || z > close_zone_z_thresh){
      local_sum += z;
      local_cnt += 1;
    }
  }

  sum_buf[tid] = local_sum;
  cnt_buf[tid] = local_cnt;
  __syncthreads();

  // block(in patch) reduction to get sum and count
  for(int s = blockDim.x / 2; s > 0; s >>= 1){
    if(tid < s){ // half of the threads is active in each step
      sum_buf[tid] += sum_buf[tid + s];
      cnt_buf[tid] += cnt_buf[tid + s];
    }
    __syncthreads();
  }

  const float threshold = (cnt_buf[0]!=0 ? (sum_buf[0] / cnt_buf[0]) : 0.0f) + cnst_lbr_margin;
  __syncthreads();

  for(unsigned int i = tid; i < n; i += blockDim.x){
    // if all_ground is true, we consider all points as ground
    const size_t glob_pt_idx = offset + i;
    metas[glob_pt_idx].ground = all_ground || (patches[glob_pt_idx].z < threshold);
    metas[glob_pt_idx].lbr = threshold; // to be able to visualize the LPR vs chosen points
  }
}

template <typename PointT>
void PatchWorkGPU<PointT>::extract_init_seeds_gpu(cudaStream_t& stream)
{
  static double lowest_h_margin_in_close_zone =
      (sensor_height_ == 0.0) ? -0.1 : adaptive_seed_selection_margin_ * sensor_height_;

  static bool _set_cnst{false};

  if (!_set_cnst) {
    cudaMemcpyToSymbol(cnst_lbr_margin, &th_seeds_, sizeof(float), 0, cudaMemcpyHostToDevice);
    _set_cnst = true;
  }

  // for patches in first zone, we only consider the points that are above the sensor height
  // for patches in other zones, all points are used to calculate mean height in patch
  // variable num of threads per patch may be useful.
  dim3 blocks(num_total_sectors_);
  size_t sm_size = NUM_THREADS_PER_PATCH * (sizeof(float) + sizeof(unsigned int));
  lbr_seed_kernel<<<blocks, NUM_THREADS_PER_PATCH, sm_size, stream>>>(
                                                                      patches_d,
                                                                      num_pts_in_patch_d,
                                                                      patch_offsets_d,
                                                                      lowest_h_margin_in_close_zone,
                                                                      zone_model_->max_ring_index_in_first_zone,
                                                                      num_min_pts_,
                                                                      metas_d
                                                                    );
}

template class PatchWorkGPU<pcl::PointXYZI>;
template class PatchWorkGPU<PointXYZILID>;
