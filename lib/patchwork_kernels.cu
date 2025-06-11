//
// Created by orcun on 25.05.2025.
//

#include "patchwork_gpu/patchwork_gpu.cuh"
#include "cub/device/device_reduce.cuh"
#include "cub/device/device_partition.cuh"

#define NUM_THREADS_PER_PATCH 128
#define WARP_SIZE 32
#define FULL_MASK 0xffffffffu

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

// Single kernel version: one block per patch with parallel reduction in shared memory
template<typename PointT>
__global__ void lbr_seed_kernel(
    const PointT* patches,
    const uint* num_pts,
    const uint* offsets,
    const double close_zone_z_thresh,
    const int max_ring_first,
    const uint min_lpr_pts_thres,
    PointMeta* metas)
{
  const int patch_idx = blockIdx.x;
  const uint n = num_pts[patch_idx];
  if (n == 0) return;

  const bool close_zone = (patch_idx < max_ring_first);
  const size_t offset = offsets[patch_idx];

  const int tid = threadIdx.x;
  extern __shared__ float shared_mem[];

  // partition whole chunk of shmem
  auto* thread_pt_z_sm = shared_mem;
  auto* iter_flag_sm = reinterpret_cast<int*>(&thread_pt_z_sm[WARP_SIZE]);
  iter_flag_sm[tid]   = -1;
  __syncthreads();

  int cum_cnt = 0;  // running total across iterations

  // how many warp scans we need to cover n items in chunks of WARP_SIZE
  int loop_times = (int)((n + WARP_SIZE - 1) / WARP_SIZE);

  int iter;
  for (iter = 0; iter < loop_times; ++iter) {
    int i = iter * WARP_SIZE + tid;
    bool local_flag = false;

    // each thread checks if its point is valid and hasn’t been counted before
    if (i < (int)n) {
      float z = patches[offset + i].z;
      bool flag = (!close_zone) || (z > close_zone_z_thresh);

      if (flag && (iter_flag_sm[tid] == -1)) {
        // Mark this thread’s slot as “counted”
        thread_pt_z_sm[tid] = z;
        iter_flag_sm[tid]    = iter;
        local_flag = true;
      }
    }

    unsigned int mask = __ballot_sync(FULL_MASK, local_flag);
    int this_iter_cnt = __popc(mask);
    //cum_cnt computed the same way in each thread
    cum_cnt += this_iter_cnt;

    // do we have enough pts
    if (cum_cnt >= (int)min_lpr_pts_thres) {
      break;
    }
  }

  // handle the case if we have more than enough pts
  uint useful_pts_num = min(cum_cnt, (int)min_lpr_pts_thres);
   /*
    EDGE CASE: n>WARP_SIZE, in close ring threads; first 15 threads of warp couldnt find valid pts, remaining 17 threads found.
    We still need 3 more. in the next iter rest of the 15 threads finds valid points and loads into ShMem.
    --> Now first use_pts_num(20) num of elements of thread_pt_z is actually ful of valid pts, but within the whole
     thread_pt_z buffer last 17 and first 3 elements must be used.
     IDEA: also keep track of in which iteration data was loaded into shmem.
     then choose pts from lower iters (from left to right in buffer) and continue with higher iters if only
     current iters were not enough to fill num_lpt_pts_thres. left to right increase rule is consistent in same iteration.
  */

  float sum_z = 0.0f;
  int used_cnt = 0;
  if (tid == 0) {
    for (int e = 0; e <= iter; ++e) {
      for (int th = 0; th < WARP_SIZE; ++th) {
        if (used_cnt == (int)useful_pts_num) break;
        if (iter_flag_sm[th] == e) {
          sum_z += thread_pt_z_sm[th];
          used_cnt++;
        }
      }
      if (used_cnt == (int)useful_pts_num) break;
    }
  }

  sum_z   = __shfl_sync(FULL_MASK, sum_z,   0);
  used_cnt = __shfl_sync(FULL_MASK, used_cnt, 0);

  float threshold = cnst_lbr_margin;
  if (used_cnt > 0) {
    threshold += (sum_z / used_cnt) ;
  }

  for (uint i = tid; i < n; i += WARP_SIZE) {
    size_t idx = offset + i;
    metas[idx].ground = (patches[idx].z < threshold);
//    metas[idx].lbr    = threshold; // only to be aable to vis. later
  }
}

template <typename PointT>
void PatchWorkGPU<PointT>::extract_init_seeds_gpu()
{
  static double lowest_h_margin_in_close_zone =
      (sensor_height_ == 0.0) ? -0.1 : adaptive_seed_selection_margin_ * sensor_height_;

  // for patches in first zone, we only consider the points that are above the sensor height
  // for patches in other zones, all points are used to calculate mean height in patch
  // variable num of threads per patch may be useful.
  dim3 blocks(num_total_sectors_);
  size_t sm_size = WARP_SIZE * (sizeof(float) + sizeof(int));
  lbr_seed_kernel<<<blocks, WARP_SIZE, sm_size, stream_>>>(
                                                    patches_d,
                                                    num_pts_in_patch_d,
                                                    patch_offsets_d,
                                                    lowest_h_margin_in_close_zone,
                                                    last_sector_1st_ring_,
                                                    num_lpr_,
                                                    metas_d
                                                  );
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
  static constexpr size_t feat_cnt= 10;
  extern __shared__ double sm_stats[]; // xx, xy, xz, yy, yz, zz, x, y, z count
  const uint patch_idx = blockIdx.x;
  const uint tid = threadIdx.x;
  const uint n = num_pts_per_patch[patch_idx];
  const PointT* patch_start = &patches[offsets[patch_idx]];
  const PointMeta* patch_metas = &metas[offsets[patch_idx]];
  float cov_mat[9]; // COL-MAJOR

  #pragma unroll
  for(size_t i=0; i<feat_cnt; ++i) {
    sm_stats[tid * feat_cnt + i] = 0.0;
  }
  __syncthreads();

  double* local_stats = &sm_stats[tid * feat_cnt];

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

  __syncthreads(); // local_stats is actuall a part of shared mem.
  for (size_t slice=blockDim.x/2; slice>0; slice>>=1) {
    if (tid < slice) {
      for(size_t stat_idx=0; stat_idx<feat_cnt; ++stat_idx) {
        sm_stats[tid * feat_cnt + stat_idx] += sm_stats[(tid + slice) * feat_cnt + stat_idx];
      }
    }
    __syncthreads();
  }

  if(tid == 0)
  {
    const double count = max(sm_stats[9], 1.0);// avoid division by zero
    const double denom_cov = (count > 1.0) ? (count - 1.0) : 1.0;
    const double denom_mean = (count >= 1.0) ? (count) : 1.0;
    const double inv_count_cov = 1.0 / denom_cov;
    const double inv_count = 1.0 / denom_mean;
    const double x_mean = sm_stats[6] * inv_count;
    const double y_mean = sm_stats[7] * inv_count;
    const double z_mean = sm_stats[8] * inv_count;

    const auto xx = static_cast<float>((sm_stats[0] - count* x_mean * x_mean) * inv_count_cov);
    const auto xy = static_cast<float>((sm_stats[1] - count* x_mean * y_mean) * inv_count_cov);
    const auto xz = static_cast<float>((sm_stats[2] - count* x_mean * z_mean) * inv_count_cov);
    const auto yy = static_cast<float>((sm_stats[3] - count* y_mean * y_mean) * inv_count_cov);
    const auto yz = static_cast<float>((sm_stats[4] - count* y_mean * z_mean) * inv_count_cov);
    const auto zz = static_cast<float>((sm_stats[5] - count* z_mean * z_mean) * inv_count_cov);

    cov_mat[0] = xx; cov_mat[3] = xy; cov_mat[6] = xz;
    cov_mat[1] = xy; cov_mat[4] = yy; cov_mat[7] = yz;
    cov_mat[2] = xz; cov_mat[5] = yz; cov_mat[8] = zz;

    // do not care about patches with insufficient points, we'll handle those later

    #pragma unroll
    for(int i = 0; i < 9; ++i) {
      cov_out[patch_idx*9 + i] = cov_mat[i];
    }
    pca_features[patch_idx].mean_ = make_float3(x_mean, y_mean, z_mean);
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
  float vx = eig_vectors_patch[0];
  float vy = eig_vectors_patch[1];
  float vz = eig_vectors_patch[2];

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
    const size_t sm_size = NUM_THREADS_PER_PATCH * 10 * sizeof(double);
    compute_patchwise_cov_mat<<<num_total_sectors_, NUM_THREADS_PER_PATCH, sm_size, stream_>>>(
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
