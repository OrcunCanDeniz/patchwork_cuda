//
// Created by orcun on 25.05.2025.
//

#include "patchwork_gpu/patchwork_gpu.cuh"
#include "cub/device/device_reduce.cuh"
#include "cub/device/device_partition.cuh"

#define NUM_THREADS_PER_PATCH 256
#define WARP_SIZE 32
#define FULL_MASK 0xffffffffu
#define COV_FT_CNT 10

__device__ __constant__ int cnst_num_sectors_per_ring[256];
__device__ __constant__ std::size_t cnst_num_sectors_per_ring_size;
__device__ __constant__ float cnst_lbr_margin;

__device__ __constant__ double cnst_uprightness_thr;
__device__ __constant__ int cnst_num_rings_of_interest;
__device__ __constant__ double cnst_elevation_thr[64];
__device__ __constant__ double cnst_sensor_height;
__device__ __constant__ double cnst_flatness_thr[64];
__device__ __constant__ bool cnst_using_global_thr;
__device__ __constant__ double cnst_global_elevation_thr;
__device__ __constant__ int cnst_min_num_pts_thr;

template <typename PointT>
void PatchWorkGPU<PointT>::set_cnst_mem()
{
  cudaMemcpyToSymbol(cnst_lbr_margin, &th_seeds_, sizeof(float));
  cudaMemcpyToSymbol(cnst_uprightness_thr, &uprightness_thr_, sizeof(double));
  cudaMemcpyToSymbol(cnst_num_rings_of_interest, &num_rings_of_interest_, sizeof(int));
  cudaMemcpyToSymbol(cnst_elevation_thr, elevation_thr_.data(), sizeof(double) * elevation_thr_.size());
  cudaMemcpyToSymbol(cnst_sensor_height, &sensor_height_, sizeof(double));
  cudaMemcpyToSymbol(cnst_flatness_thr, flatness_thr_.data(), sizeof(double) * flatness_thr_.size());
  cudaMemcpyToSymbol(cnst_using_global_thr, &using_global_thr_, sizeof(bool));
  cudaMemcpyToSymbol(cnst_global_elevation_thr, &global_elevation_thr_, sizeof(double));
  cudaMemcpyToSymbol(cnst_min_num_pts_thr, &num_min_pts_, sizeof(int));
}

__global__ void lbr_seed_kernel(
    const float* pt_z,
    const uint* num_pts,
    const uint* offsets,
    const double close_zone_z_thresh,
    const uint max_ring_first,
    const int max_lpr_pts_thres,
    float* thres_d)
{
  const int patch_idx = blockIdx.x;
  const uint n = num_pts[patch_idx];
  if (n == 0) return;

  const bool close_zone = (patch_idx < max_ring_first);
  const size_t offset = offsets[patch_idx];

  const int tid = threadIdx.x;

  int local_iter{-1};
  float local_z{0};

  int cum_cnt = 0;  // running total across iterations

  // how many warp scans we need to cover n items in chunks of WARP_SIZE
  int loop_times = (int)((n + WARP_SIZE - 1) / WARP_SIZE);

  int iter;
  for (iter = 0; iter < loop_times; ++iter) {
    int i = iter * WARP_SIZE + tid;
    bool local_flag = false;

    if (i < (int)n) {
      float z = pt_z[offset + i];
      const bool flag = (!close_zone) || (z > close_zone_z_thresh);
      // each thread checks if its point is valid and it's not already holding a value
      if (flag && (local_iter == -1)) {
        // Mark this thread’s slot as “counted”
        local_z = z;
        local_iter = iter;
        local_flag = true;
      }
    }

    unsigned int mask = __ballot_sync(FULL_MASK, local_flag);
    int this_iter_cnt = __popc(mask);
    //cum_cnt computed the same way in each thread
    cum_cnt += this_iter_cnt;

    // do we have enough pts
    if (cum_cnt >= max_lpr_pts_thres) {
      break;
    }
  }

  // handle the case if we have more than enough pts
  const uint useful_pts_num = min(cum_cnt, max_lpr_pts_thres);
  /*
   EDGE CASE: n>WARP_SIZE, in close ring threads; first 15 threads of warp couldnt find valid pts, remaining 17 threads found.
   We still need 3 more. in the next iter rest of the 15 threads finds valid points and loads into ShMem.
   --> Now, first use_pts_num(20) num of elements of thread_pt_z is actually full of valid pts, but within the whole
    thread_pt_z buffer last 17 and first 3 elements must be used.
    IDEA: also keep track of in which iteration data was loaded into shmem.
    then choose pts from lower iters (from left to right in buffer) and continue with higher iters if only
    current iters were not enough to fill num_lpt_pts_thres. left to right increase rule is consistent in same iteration.
 */

  float sum_z = 0.0f;
  int used_cnt = 0;
  for (int e = 0; e <= iter; ++e) {
    uint curr_iter_mask = __ballot_sync(FULL_MASK, e==local_iter);  // which threads holds zs from iter e
    const int curr_iter_cnt = __popc(curr_iter_mask); // how many valid zs
    const bool overflows = useful_pts_num < (used_cnt + curr_iter_cnt); // if adding all zs from e overflow total needed points
    uint numel_contrib{0};
    if (overflows){
      // subset from current iter must be used
      const uint subset_size = useful_pts_num - used_cnt;
      const uint subset_mask = (1u<<(subset_size)) - 1; // mask the subset that'll be used for e
      numel_contrib = subset_size;
      curr_iter_mask = subset_mask ;
    } else {
      numel_contrib = curr_iter_cnt;
    }
    // count+accum by warp reduction
    float iter_sum = ((curr_iter_mask >> tid) & 1) ? local_z : 0;
    for (int offset = 16; offset > 0; offset >>= 1)
      iter_sum += __shfl_down_sync(FULL_MASK, iter_sum, offset);
    iter_sum = __shfl_sync(FULL_MASK, iter_sum, 0);
    sum_z += iter_sum;
    used_cnt += numel_contrib;
  }

  if(tid==0){
    float threshold = cnst_lbr_margin + (used_cnt > 0 ? sum_z / used_cnt : 0);
    thres_d[patch_idx] = threshold;
  }
}

__global__ void seed_mark_kernel(
    const float* z_buffer_d,
    const float* thres_d,
    PointMeta* metas_d,
    const uint num_patched_pts)
{
  const uint pt_idx = blockDim.x * blockIdx.x + threadIdx.x;
  if ( pt_idx >= num_patched_pts) return;
  const auto patch_idx = metas_d[pt_idx].lin_sec_idx;
  const float seed_thr = thres_d[patch_idx];
  const float pt_z = z_buffer_d[pt_idx];
  metas_d[pt_idx].ground = pt_z < seed_thr;
}

template <typename PointT>
void PatchWorkGPU<PointT>::extract_init_seeds_gpu()
{
  static double lowest_h_margin_in_close_zone =
      (sensor_height_ == 0.0) ? -0.1 : adaptive_seed_selection_margin_ * sensor_height_;

  // for patches in first zone, we only consider the points that are above the sensor height
  // for patches in other zones, all points are used to calculate mean height in patch
  // variable num of threads per patch may be useful.
  lbr_seed_kernel<<<num_total_sectors_, WARP_SIZE, 0, stream_>>>(
                                                    patched_z_d,
                                                    num_pts_in_patch_d,
                                                    patch_offsets_d,
                                                    lowest_h_margin_in_close_zone,
                                                    last_sector_1st_ring_,
                                                    num_lpr_,
                                                    patch_seed_thr_d
                                                  );


  const dim3 blocks(divup(*num_patched_pts_h, NUM_THREADS_PER_PATCH));
  seed_mark_kernel<<<blocks,NUM_THREADS_PER_PATCH,0, stream_ >>>(patched_z_d, patch_seed_thr_d,
                                                       metas_d, *num_patched_pts_h);

}

template<typename PointT>
__global__ void compute_patchwise_cov_mat (const PointT* patches,
                                            const uint* num_pts_per_patch,
                                            const uint* offsets,
                                            float* cov_out,
                                            PointMeta* metas,
                                            PCAFeature* pca_features
                                          )
{
  float local_stats[10]= {0}; // xx, xy, xz, yy, yz, zz, x, y, z count
  const uint patch_idx = blockIdx.x;
  const uint tid = threadIdx.x;
  const uint lane_id = tid & (WARP_SIZE-1);
  const uint n = num_pts_per_patch[patch_idx];
  const PointT* patch_start = &patches[offsets[patch_idx]];
  const PointMeta* patch_metas = &metas[offsets[patch_idx]];

  // each thread accumulate to its registers
  for (size_t i=tid; i<n; i+=blockDim.x) {
    const bool is_ground = patch_metas[i].ground;
    const PointT& pt = patch_start[i];

    local_stats[0] += pt.x * pt.x * is_ground;
    local_stats[1] += pt.x * pt.y * is_ground;
    local_stats[2] += pt.x * pt.z * is_ground;
    local_stats[3] += pt.y * pt.y * is_ground;
    local_stats[4] += pt.y * pt.z * is_ground;
    local_stats[5] += pt.z * pt.z * is_ground;
    local_stats[6] += pt.x * is_ground;
    local_stats[7] += pt.y * is_ground;
    local_stats[8] += pt.z * is_ground;
    local_stats[9] += is_ground;
  }
  __syncthreads();

  // combine warps' threads' results
  for (size_t offset=WARP_SIZE/2; offset>0; offset>>=1) {
    #pragma unroll
    for(int f=0; f<COV_FT_CNT; ++f){
      local_stats[f] += __shfl_down_sync(FULL_MASK, local_stats[f], offset);
    }
  }

  constexpr uint num_warps = NUM_THREADS_PER_PATCH/WARP_SIZE;
  const uint warp_id = tid >> 5; // tid/32 = tid/WARP_SIZE
  __shared__ float warp_results[num_warps * COV_FT_CNT];
  float* this_warp_res = &warp_results[warp_id*COV_FT_CNT];

  // each warp's leader thread loads warp results to sm
  if (lane_id == 0)
  {
    #pragma unroll
    for(int f=0; f<COV_FT_CNT; ++f) this_warp_res[f] = local_stats[f];
  }
  __syncthreads();

  if(warp_id==0) {
    float final_warp_local[COV_FT_CNT];
    // each thread in final warp loads from sm to registers
    if (lane_id < num_warps) {
      #pragma unroll
      for (int f = 0; f < COV_FT_CNT; ++f) {
        final_warp_local[f] = warp_results[lane_id * COV_FT_CNT + f];
      }
    } else {
      #pragma unroll
      for (int f = 0; f < COV_FT_CNT; ++f) final_warp_local[f] = 0;  // set empty warps to zero
    }

    // leader warp reduce results of multiple warps in block
    for (size_t offset = num_warps >> 1; offset > 0; offset >>= 1) {
      #pragma unroll
      for (int f = 0; f < COV_FT_CNT; ++f) {
        final_warp_local[f] += __shfl_down_sync(FULL_MASK, final_warp_local[f], offset);
      }
    }

    if (lane_id == 0) {
      const float count = max(final_warp_local[9], 1.0);  // avoid division by zero
      const float denom_cov = (count > 1.0) ? (count - 1.0) : 1.0;
      const float denom_mean = (count >= 1.0) ? (count) : 1.0;
      const float inv_count_cov = 1.0 / denom_cov;
      const float inv_count = 1.0 / denom_mean;
      const float x_mean = final_warp_local[6] * inv_count;
      const float y_mean = final_warp_local[7] * inv_count;
      const float z_mean = final_warp_local[8] * inv_count;

      const float xx = (final_warp_local[0] - count * x_mean * x_mean) * inv_count_cov;
      const float xy = (final_warp_local[1] - count * x_mean * y_mean) * inv_count_cov;
      const float xz = (final_warp_local[2] - count * x_mean * z_mean) * inv_count_cov;
      const float yy = (final_warp_local[3] - count * y_mean * y_mean) * inv_count_cov;
      const float yz = (final_warp_local[4] - count * y_mean * z_mean) * inv_count_cov;
      const float zz = (final_warp_local[5] - count * z_mean * z_mean) * inv_count_cov;

      float* co = cov_out + patch_idx*9;
      co[0]=xx; co[1]=xy; co[2]=xz;
      co[3]=xy; co[4]=yy; co[5]=yz;
      co[6]=xz; co[7]=yz; co[8]=zz;

      // do not care about patches with insufficient points, we'll handle those later

      pca_features[patch_idx].mean_ = make_float3(x_mean, y_mean, z_mean);
    }
  }
}

__global__ void set_patch_pca_features(float* eig_vects,
                                       float* eig_vals,
                                       PCAFeature* pca_features,
                                       const float th_dist)
{
  const uint patch_idx = blockIdx.x;
  PCAFeature pca_feature = pca_features[patch_idx];

  // change the order of eigen values to match OG implementation, just to make it easier to reimplement
  // cusolver sorts them in ascending order.
  pca_feature.singular_values_ = make_float3(static_cast<float>(eig_vals[patch_idx * 3 + 2]),
                                             static_cast<float>(eig_vals[patch_idx * 3 + 1]),
                                             static_cast<float>(eig_vals[patch_idx * 3 ]));

  auto inv_sing_val = 1/ pca_feature.singular_values_.x;
  pca_feature.linearity_ =
      (pca_feature.singular_values_.x - pca_feature.singular_values_.y) * inv_sing_val;
  pca_feature.planarity_ =
      (pca_feature.singular_values_.y - pca_feature.singular_values_.z) * inv_sing_val;

  // 1st vect is the one with least eig val. thus plane normal.
  float* eig_vectors_patch = &eig_vects[patch_idx * 9];
  // eig vectors are stored col-major
  const float vx = eig_vectors_patch[0];
  const float vy = eig_vectors_patch[1];
  const float vz = eig_vectors_patch[2];

  int inv_vect = (vz < 0.0f) ? -1 : 1;
  pca_feature.normal_ = make_float3( vx * inv_vect,
                                     vy * inv_vect,
/* z of normal vector must be pos */ vz * inv_vect);

  pca_feature.d_ = -(pca_feature.normal_.x * pca_feature.mean_.x +
                      pca_feature.normal_.y * pca_feature.mean_.y +
                      pca_feature.normal_.z * pca_feature.mean_.z);

  pca_feature.th_dist_d_ = th_dist - pca_feature.d_;
  pca_features[patch_idx] = pca_feature;
}
template <typename PointT>
__global__ void filter_by_dist2plane(const PointT* patches,
                                      const PCAFeature* pca_features,
                                      PointMeta* metas,
                                     const uint num_patched_pts)
{
  const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid>= num_patched_pts) return;

  const PointMeta meta = metas[tid];
  const uint& patch_idx = meta.lin_sec_idx;
  const PCAFeature& feat = pca_features[patch_idx];

  const PointT& pt = patches[tid];

  const float dist = feat.normal_.x * pt.x +
                      feat.normal_.y * pt.y +
                      feat.normal_.z * pt.z ;

  metas[tid].ground = dist < feat.th_dist_d_;
}


template <typename PointT>
void PatchWorkGPU<PointT>::fit_regionwise_planes_gpu()
// stream selection enforced intentionally, cuSolver is binded with specific stream at somewhere else
{
  static bool work_d_alloced{false};
  static int lwork{0};

  for(size_t i=0; i<num_iter_; ++i) {
    // compute cov matrices for each patch
    compute_patchwise_cov_mat<<<num_total_sectors_, NUM_THREADS_PER_PATCH, 0, stream_>>>(
        patches_d, num_pts_in_patch_d, patch_offsets_d, cov_mats_d, metas_d, pca_features_d);

    // run PCA on each patch -> eigenvector w/ least eig, val is the normal
    // cov mat is always positive-semidefinite, so, eigen vectors = singular vectors
    // we can do eigen decomp instead of SVD minor difference from OG implementation

    if(!work_d_alloced)
    {
      // allocate workspace for cuSolver
      cudaStreamSynchronize(stream_);
      CUSOLVER_CHECK(cusolverDnSsyevjBatched_bufferSize(cusolverH,
                                                        CUSOLVER_EIG_MODE_VECTOR,
                                                        CUBLAS_FILL_MODE_UPPER,
                                                        3,
                                                        cov_mats_d,
                                                        3,
                                                        eigen_vals_d,
                                                        &lwork,
                                                        syevj_params,
                                                        num_total_sectors_));

      CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&work_d), sizeof(float) * lwork, stream_));
      work_d_alloced = true;
    }

    CUSOLVER_CHECK(cusolverDnSsyevjBatched(cusolverH,
                                           CUSOLVER_EIG_MODE_VECTOR,
                                           CUBLAS_FILL_MODE_UPPER,
                                           3,
                                           cov_mats_d,
                                           3,
                                           eigen_vals_d,
                                           work_d,
                                           lwork,
                                           eig_info_d,
                                           syevj_params,
                                           num_total_sectors_));
    // covariance mats. in cov_mats_d are now eigen vectors

    // compute patchwise PCA features
    set_patch_pca_features<<<num_total_sectors_, 1, 0, stream_>>>(cov_mats_d, eigen_vals_d,
                                                                  pca_features_d, th_dist_);
    //    // choose points by their dist to estimated plane
    dim3 blocks(divup(*num_patched_pts_h, NUM_THREADS_PER_PATCH));
    filter_by_dist2plane<<<blocks, NUM_THREADS_PER_PATCH, 0, stream_>>>(
        patches_d, pca_features_d, metas_d, *num_patched_pts_h);
  }

  auto err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    std::cerr << "CUDA error in fit_regionwise_planes_gpu: " << cudaGetErrorString(err) << std::endl;
  }
}

__device__ uint8_t compute_ground_likelihood_estimation_statusv2(
    const int ring_idx,
    const double z_vec,
    const double z_elevation,
    const double surface_variable,
    const uint num_pts_in_patch) {
  const bool is_too_few_pts = num_pts_in_patch < cnst_min_num_pts_thr;
  const bool is_too_tilted = z_vec < cnst_uprightness_thr;
  const bool close_ring_flag = (ring_idx < cnst_num_rings_of_interest);
  const bool elev_thr_flag = z_elevation > (-cnst_sensor_height + cnst_elevation_thr[ring_idx]);
  const bool flatness_flag = cnst_flatness_thr[ring_idx] > surface_variable;
  const bool glob_thr_flag = cnst_using_global_thr && (z_elevation > cnst_global_elevation_thr);

  const bool is_upright_enough1 = !is_too_tilted && close_ring_flag && !elev_thr_flag;
  const bool is_flat_enough = !is_too_tilted && close_ring_flag && elev_thr_flag && flatness_flag;
  const bool is_too_high_elev =
      !is_too_tilted && close_ring_flag && elev_thr_flag && !flatness_flag;
  const bool is_glob_too_high_elev = !is_too_tilted && !close_ring_flag && glob_thr_flag;
  const bool is_upright_enough2 = !is_too_tilted && !close_ring_flag && !glob_thr_flag;
  bool states[6] = {
      is_too_few_pts, // 0
      !is_too_few_pts && is_too_tilted, // 1
      !is_too_few_pts && is_flat_enough,     // 2
      !is_too_few_pts && is_too_high_elev,   // 3
      !is_too_few_pts && (is_upright_enough1 || is_upright_enough2),  // 4
      !is_too_few_pts && is_glob_too_high_elev, // 5
  };
  uint8_t sum= 0;
  for(int i = 0; i < 5; ++i) {
    sum += states[i] * i;
    if (states[i]) break;
  }
  return sum;
}
__global__ void compute_patch_states(const uint* num_pts_in_patch,
                                     PCAFeature* pca_features,
                                     PatchState* patch_states)
{
  // This kernel is not used in the current implementation, but can be used to compute additional patch features
  // if needed in the future.
  const uint patch_idx = blockIdx.x;
  const uint n = num_pts_in_patch[patch_idx];
  PCAFeature& pca_feat = pca_features[patch_idx];
  const float min_singular_val = min(
      min(pca_feat.singular_values_.x, pca_feat.singular_values_.y), pca_feat.singular_values_.z);

  const double ground_z_vec = fabs(pca_feat.normal_.z);
  const double ground_z_elevation = pca_feat.mean_.z;
  const double surface_variable = min_singular_val / (pca_feat.singular_values_.x +
                                                      pca_feat.singular_values_.y +
                                                      pca_feat.singular_values_.z);
  auto ring_idx = ring_sec_idx_from_lin_idx(patch_idx).x;

  const uint8_t state = compute_ground_likelihood_estimation_statusv2(ring_idx,ground_z_vec,
                                                                       ground_z_elevation,
                                                                       surface_variable,n);
  patch_states[patch_idx] = static_cast<PatchState>(state);
}

__global__ void set_groundness(
    const uint num_patched_pts,
    PointMeta* metas,
    PatchState* patch_states)
{
  const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_patched_pts) return;

  const PointMeta meta = metas[tid];
  const uint& patch_idx = meta.lin_sec_idx;
  const PatchState feat = patch_states[patch_idx];

  // all_ground override : few_points,
  // all_NON_ground override : too_tilted, globally_too_high_elev, too_high_elev
  // as_is : flat_enough, upright_enough
  // no need to check if (state == FLAT_ENOUGH or state == UPRIGHT_ENOUGH)
  // these states keeps the groundness as it is. below expression will hopefully handle all cases already.
  const bool all_ground_override = (feat == PatchState::FEW_PTS);
  const bool all_NONground_override =
      (feat == PatchState::TOO_TILTED || feat == PatchState::GLOB_TOO_HIGH_ELEV ||
       feat == PatchState::TOO_HIGH_ELEV);

//  const bool ground_decision = max(
//        min(meta.ground + all_ground_override,1) - all_NONground_override, 0);
/* Above exoression is equivalent to the following truth table:
  meta_ground | all_ground_override | all_NONground_override | ground_decision
  False       | False               | False                  | False
  False       | False               | True                   | False
  False       | True                | False                  | True
  True        | False               | False                  | True
  True        | False               | True                   | False
  True        | True                | False                  | True
*/

  if (all_ground_override)
  {
    metas[tid].ground = true;
  } else if (all_NONground_override) {
    metas[tid].ground = false;
  } else{
    metas[tid].ground = metas[tid].ground;
  }
 }

template <typename PointT>
void PatchWorkGPU<PointT>::finalize_groundness_gpu()
{
  compute_patch_states<<<num_total_sectors_, 1, 0, stream_>>>(num_pts_in_patch_d,
                                                              pca_features_d,
                                                              patch_states_d);

  dim3 blocks(divup(*num_patched_pts_h, NUM_THREADS_PER_PATCH));
  set_groundness<<<blocks, NUM_THREADS_PER_PATCH, 0, stream_>>>( *num_patched_pts_h,
                                                                  metas_d,
                                                                  patch_states_d
                                                                );
}




//template class PatchWorkGPU<pcl::PointXYZI>;
template class PatchWorkGPU<PointXYZILID>;
