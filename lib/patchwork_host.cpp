//
// Created by orcun on 25.05.2025.
//

#include "patchwork_gpu/patchwork_gpu.cuh"

template<typename PointT>
PatchWorkGPU<PointT>::PatchWorkGPU(ros::NodeHandle *nh)
{
  // Init ROS related
  ROS_INFO("Inititalizing PatchWork...");
  condParam(nh, "verbose", verbose_, false);
  condParam(nh, "sensor_height", sensor_height_, 1.723, "");
  condParam(nh, "sensor_model", sensor_model_, std::string("HDL-64E"), "");

  condParam(nh, "ATAT/ATAT_ON", ATAT_ON_, false);
  condParam(nh, "ATAT/max_r_for_ATAT", max_r_for_ATAT_, 5.0);
  condParam(nh, "ATAT/num_sectors_for_ATAT", num_sectors_for_ATAT_, 20);
  condParam(nh, "ATAT/noise_bound", noise_bound_, 0.2);

  condParam(nh, "num_iter", num_iter_, 3);
  condParam(nh, "num_lpr", num_lpr_, 20);
  condParam(nh, "num_min_pts", num_min_pts_, 10);
  condParam(nh, "th_seeds", th_seeds_, 0.4f);
  condParam(nh, "th_dist", th_dist_, 0.3);
  condParam(nh, "max_r", max_range_, 80.0);
  condParam(nh, "min_r", min_range_,
            2.7);  // It should cover the body size of the car.
  condParam(nh, "uniform/num_rings", num_rings_, 30);
  condParam(nh, "uniform/num_sectors", num_sectors_, 108);
  condParam(nh, "uprightness_thr", uprightness_thr_,
            0.5);  // The larger, the more strict
  // The larger, the more soft
  condParam(nh, "adaptive_seed_selection_margin", adaptive_seed_selection_margin_, -1.1);

  // It is not in the paper
  // It is also not matched our philosophy, but it is employed to reject some
  // FPs easily & intuitively. For patchwork, it is only applied on Z3 and Z4
  condParam(nh, "using_global_elevation", using_global_thr_, true);
  condParam(nh, "global_elevation_threshold", global_elevation_thr_, 0.0);

  if (using_global_thr_) {
    std::cout << "\033[1;33m[Warning] Global elevation threshold is turned on :"
              << global_elevation_thr_ << "\033[0m" << std::endl;
  } else {
    std::cout << "Global thr. is not in use" << std::endl;
  }

  ROS_INFO("Sensor Height: %f", sensor_height_);
  ROS_INFO("Num of Iteration: %d", num_iter_);
  ROS_INFO("Num of LPR: %d", num_lpr_);
  ROS_INFO("Num of min. points: %d", num_min_pts_);
  ROS_INFO("Seeds Threshold: %f", th_seeds_);
  ROS_INFO("Distance Threshold: %f", th_dist_);
  ROS_INFO("Max. range:: %f", max_range_);
  ROS_INFO("Min. range:: %f", min_range_);
  ROS_INFO("adaptive_seed_selection_margin: %f", adaptive_seed_selection_margin_);

  // CZM denotes 'Concentric Zone Model'. Please refer to our paper
  // 2024.07.28. I feel `num_zones_`, `num_sectors_each_zone_`,
  // num_rings_each_zone_` are rarely fine-tuned. So I've decided to provide
  // predefined parameter sets for sensor types
  condParam(nh, "czm/elevation_thresholds", elevation_thr_, {0.523, 0.746, 0.879, 1.125});
  condParam(nh, "czm/flatness_thresholds", flatness_thr_, {0.0005, 0.000725, 0.001, 0.001});

  ROS_INFO("\033[1;32mUprightness\33[0m threshold: %f", uprightness_thr_);
  ROS_INFO("\033[1;32mElevation\33[0m thresholds: %f %f %f %f",
           elevation_thr_[0],
           elevation_thr_[1],
           elevation_thr_[2],
           elevation_thr_[3]);
  ROS_INFO("\033[1;32mFlatness\033[0m thresholds: %f %f %f %f",
           flatness_thr_[0],
           flatness_thr_[1],
           flatness_thr_[2],
           flatness_thr_[3]);

  zone_model_ = std::make_unique<ConcentricZoneModelGPU<PointT>>(sensor_model_, sensor_height_,
                                                                 min_range_, max_range_, max_pts_in_cld_);
  ROS_INFO("Num. zones: %zu", zone_model_->num_zones_);

  float tmp_max_pts_in_cld_ = 0;
  condParam(nh, "max_pts_in_cloud", tmp_max_pts_in_cld_, 300000.f);
  max_pts_in_cld_ = static_cast<uint32_t>(tmp_max_pts_in_cld_);

  // It equals to elevation_thr_.size()/flatness_thr_.size();
  num_rings_of_interest_ = elevation_thr_.size();

  condParam(nh, "visualize", visualize_, true);
  condParam<std::string>(nh, "frame_patchwork", frame_patchwork, frame_patchwork);


  PlanePub = nh->advertise<jsk_recognition_msgs::PolygonArray>("/gpf/plane", 100);
  RevertedCloudPub = nh->advertise<sensor_msgs::PointCloud2>("/revert_pc", 100);
  RejectedCloudPub = nh->advertise<sensor_msgs::PointCloud2>("/reject_pc", 100);

  const auto &num_sectors_each_zone_ = zone_model_->sensor_config_.num_sectors_for_each_zone_;
  sector_sizes_ = {2 * M_PI / num_sectors_each_zone_.at(0),
                   2 * M_PI / num_sectors_each_zone_.at(1),
                   2 * M_PI / num_sectors_each_zone_.at(2),
                   2 * M_PI / num_sectors_each_zone_.at(3)};

  init_cuda();
  last_sector_1st_ring_ = std::accumulate(zone_model_->num_sectors_per_ring_.begin(),
                                              zone_model_->num_sectors_per_ring_.begin() + zone_model_->max_ring_index_in_first_zone, 0);


  std::cout << "INITIALIZATION COMPLETE" << std::endl;
}


template<typename PointT>
void PatchWorkGPU<PointT>::init_cuda()
{

  CUDA_CHECK(cudaMalloc((void**)&cloud_in_d_, sizeof(PointT) * max_pts_in_cld_));

  CUDA_CHECK(cudaStreamCreate(&stream_));
  CUDA_CHECK(cudaStreamCreate(&streamd2h_));
  CUDA_CHECK(cudaStreamCreate(&streamh2d_));

  CUDA_CHECK(cudaEventCreate(&cuEvent_start));
  CUDA_CHECK(cudaEventCreate(&cuEvent_stop));

  // allocate memory for patches
  patches_size = max_pts_in_cld_ * sizeof(PointT);
  CUDA_CHECK(cudaMalloc((void**)&patches_d, patches_size));

  num_total_sectors_ = std::accumulate(zone_model_->num_sectors_per_ring_.begin(),
                                        zone_model_->num_sectors_per_ring_.end(), 0);

  num_pts_in_patch_size = (num_total_sectors_) * sizeof(uint);
  CUDA_CHECK(cudaMalloc((void**)&num_pts_in_patch_d, num_pts_in_patch_size));
  CUDA_CHECK(cudaMalloc((void**)&patch_states_d, sizeof(PatchState)* num_total_sectors_));
  CUDA_CHECK(cudaMalloc((void**)&patch_offsets_d, num_pts_in_patch_size + sizeof(uint)));
  CUDA_CHECK(cudaMallocHost((void**)&patch_offsets_h, num_pts_in_patch_size + sizeof(uint)));

  CUDA_CHECK(cudaMalloc((void**)&in_metas_d, sizeof(PointMeta) * max_pts_in_cld_));
  CUDA_CHECK(cudaMalloc((void**)&metas_d, sizeof(PointMeta) * max_pts_in_cld_));

  CUDA_CHECK(cudaMalloc((void**)&cov_mats_d, sizeof(float)* 3 * 3 * num_total_sectors_));
  CUDA_CHECK(cudaMalloc((void**)&pca_features_d, num_total_sectors_ * sizeof(PCAFeature)));

  // cusolver buffers
  CUDA_CHECK(cudaMalloc((void**)&eigen_vals_d, num_total_sectors_ * 3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&eig_info_d, num_total_sectors_ * sizeof(int)));

  CUDA_CHECK(cudaMalloc((void**)&patch_seed_thr_d, num_total_sectors_ * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&patched_z_d, max_pts_in_cld_ * sizeof(float)));

  CUDA_CHECK(cudaMallocHost((void**)&num_patched_pts_h, sizeof(uint)));

  CUDA_CHECK(cudaMallocHost((void**)&packed_pts_out_h, sizeof(PointT) * max_pts_in_cld_));
  CUDA_CHECK(cudaMallocHost((void**)&metas_h, sizeof(PointMeta) * max_pts_in_cld_));

  setup_cusolver();
  set_cnst_mem();
  reset_buffers();
}
template<typename PointT>
void PatchWorkGPU<PointT>::setup_cusolver()
{
  CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
  CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream_));
  CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params));
  CUSOLVER_CHECK(cusolverDnXsyevjSetTolerance(syevj_params, 1.e-7));
  CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(syevj_params, 50));
  CUSOLVER_CHECK(cusolverDnXsyevjSetSortEig(syevj_params, 1));
}


template<typename PointT>
void PatchWorkGPU<PointT>::reset_buffers(cudaStream_t stream)
{
  CUDA_CHECK(cudaMemsetAsync(cloud_in_d_, 0, sizeof(PointT) * max_pts_in_cld_, stream));
  CUDA_CHECK(cudaMemsetAsync(patches_d, 0, patches_size, stream));
  CUDA_CHECK(cudaMemsetAsync(metas_d, 0, sizeof(PointMeta) * max_pts_in_cld_, stream));
  CUDA_CHECK(cudaMemsetAsync(num_pts_in_patch_d, 0, num_pts_in_patch_size, stream));
  CUDA_CHECK(cudaMemsetAsync(patch_states_d, 0, sizeof(PatchState) * num_total_sectors_, stream));
  CUDA_CHECK(cudaMemsetAsync(patch_offsets_d, 0, num_pts_in_patch_size+ sizeof(uint), stream));
  CUDA_CHECK(cudaMemsetAsync(cov_mats_d, 0, sizeof(float) * 3 * 3 * num_total_sectors_, stream));
  CUDA_CHECK(cudaMemsetAsync(pca_features_d, 0, num_total_sectors_ * sizeof(PCAFeature), stream));
  CUDA_CHECK(cudaMemsetAsync(eigen_vals_d, 0, num_total_sectors_ * 3 * sizeof(float), stream));
  CUDA_CHECK(cudaMemsetAsync(eig_info_d, 0, num_total_sectors_ * sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(patch_seed_thr_d, 0, num_total_sectors_ * sizeof(float), stream));
  CUDA_CHECK(cudaMemsetAsync(patched_z_d, 0, max_pts_in_cld_ * sizeof(float), stream));
  *num_patched_pts_h = 0;
}


template <typename PointT>
void PatchWorkGPU<PointT>::to_pcl(pcl::PointCloud<PointT>* ground,
                                  pcl::PointCloud<PointT>* nonground,
                                  const bool with_lpr)
{
  cudaMemcpyAsync(packed_pts_out_h, patches_d, sizeof(PointT) * (*num_patched_pts_h),
                  cudaMemcpyDeviceToHost, streamd2h_);
  ground->reserve(*num_patched_pts_h);
  nonground->reserve(*num_patched_pts_h);
  cudaStreamSynchronize(stream_);
  cudaMemcpy(metas_h, metas_d, sizeof(PointMeta) * (*num_patched_pts_h),
                  cudaMemcpyDeviceToHost); // dont synch for this op yet.
  // parse into ground pcl cloud
  for (size_t i=0; i<(*num_patched_pts_h); i++)
  {
    static PointT pt_lpr;
    PointT& pt = packed_pts_out_h[i];
    if(with_lpr)
    {
      pt_lpr = pt;
//      pt_lpr.z = metas_h[i].lbr;
      pt_lpr.intensity = 100;
      pt.intensity = 0;
    }
    if (metas_h[i].ground)
    {
      ground->push_back(pt);
      if(with_lpr) ground->push_back(pt_lpr);
    } else {
      nonground->push_back(pt);
      if(with_lpr) nonground->push_back(pt_lpr);
    }
  }
}

template<typename PointT>
void PatchWorkGPU<PointT>::estimate_ground(pcl::PointCloud<PointT>* cloud_in,
                                           pcl::PointCloud<PointT>* ground,
                                           pcl::PointCloud<PointT>* nonground,
                                           float* time_taken)
{
  // TODO: sensor height estimation is not implemented yet
  reset_buffers();

  CUDA_CHECK(cudaMemcpyAsync(cloud_in_d_, cloud_in->points.data(), cloud_in->points.size() * sizeof(PointT),
                             cudaMemcpyHostToDevice, stream_));
  ground->clear();
  nonground->clear();
  // mark the start of processing
  cudaEventRecord(cuEvent_start,stream_);

  bool ret = zone_model_->create_patches_gpu(cloud_in_d_, cloud_in->points.size(),
                                             num_pts_in_patch_d,  in_metas_d, metas_d,
                                             patch_offsets_d, num_total_sectors_,
                                             patches_d, *num_patched_pts_h,
                                             patched_z_d, stream_);
  if(!ret)
  {
    throw std::runtime_error("Failed to launch create patches kernel.");
  }
  // at this point we have only valid points, compacted, assigned to patches.
  // they're sorted among their patch by ascending z. respective meta for a point can be accessed
  // using the same buffer idx. patches[idx] -> meta[idx]
  extract_init_seeds_gpu();
  fit_regionwise_planes_gpu();
  finalize_groundness_gpu();

  // mark the end of processing
  cudaEventRecord(cuEvent_stop,stream_);
  cudaEventSynchronize(cuEvent_stop);
  cudaEventElapsedTime(time_taken, cuEvent_start, cuEvent_stop);
  to_pcl(ground,nonground);
}

template class PatchWorkGPU<PointXYZILID>;

