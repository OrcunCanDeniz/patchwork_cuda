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

  poly_list_.header.frame_id = frame_patchwork;
  poly_list_.polygons.reserve(130000);

  reverted_points_by_flatness_.reserve(NUM_HEURISTIC_MAX_PTS_IN_PATCH);

  PlanePub = nh->advertise<jsk_recognition_msgs::PolygonArray>("/gpf/plane", 100);
  RevertedCloudPub = nh->advertise<sensor_msgs::PointCloud2>("/revert_pc", 100);
  RejectedCloudPub = nh->advertise<sensor_msgs::PointCloud2>("/reject_pc", 100);

  const auto &num_sectors_each_zone_ = zone_model_->sensor_config_.num_sectors_for_each_zone_;
  sector_sizes_ = {2 * M_PI / num_sectors_each_zone_.at(0),
                   2 * M_PI / num_sectors_each_zone_.at(1),
                   2 * M_PI / num_sectors_each_zone_.at(2),
                   2 * M_PI / num_sectors_each_zone_.at(3)};

//  initialize(regionwise_patches_);
  init_cuda();
  std::cout << "INITIALIZATION COMPLETE" << std::endl;
}


template<typename PointT>
void PatchWorkGPU<PointT>::init_cuda()
{

  CUDA_CHECK(cudaMalloc((void**)&cloud_in_d_, sizeof(PointT) * max_pts_in_cld_));

  cudaStreamCreate(&stream_);
  cudaStreamCreate(&streamd2h_);
  cudaStreamCreate(&streamh2d_);

  // allocate memory for patches
  patches_size = max_pts_in_cld_ * sizeof(PointT);
  CUDA_CHECK(cudaMalloc((void**)&patches_d, patches_size));

  num_total_sectors_ = std::accumulate(zone_model_->num_sectors_per_ring_.begin(),
                                        zone_model_->num_sectors_per_ring_.end(), 0);

  num_pts_in_patch_size = num_total_sectors_ * sizeof(uint);
  CUDA_CHECK(cudaMalloc((void**)&num_pts_in_patch_d, num_pts_in_patch_size));
  CUDA_CHECK(cudaMalloc((void**)&patch_states_d, sizeof(PatchState)* num_total_sectors_));
  CUDA_CHECK(cudaMalloc((void**)&patch_offsets_d, num_pts_in_patch_size));
  CUDA_CHECK(cudaMallocHost((void**)&patch_offsets_h, num_pts_in_patch_size));

  CUDA_CHECK(cudaMalloc((void**)&in_metas_d, sizeof(PointMeta) * max_pts_in_cld_));
  CUDA_CHECK(cudaMalloc((void**)&metas_d, sizeof(PointMeta) * max_pts_in_cld_));

  CUDA_CHECK(cudaMalloc((void**)&cov_mats_d, sizeof(float)* 3 * 3 * num_total_sectors_));
  CUDA_CHECK(cudaMalloc((void**)&pca_features_d, num_total_sectors_ * sizeof(PCAFeature)));

  // cusolver buffers
  CUDA_CHECK(cudaMalloc((void**)&eigen_vals_d, num_total_sectors_ * 3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&eig_info_d, num_total_sectors_ * sizeof(int)));
  eig_info_h.resize(num_total_sectors_);
  CUDA_CHECK(cudaMallocHost((void**)&eig_info_h, num_total_sectors_ * sizeof(int)));

  CUDA_CHECK(cudaMallocHost((void**)&num_patched_pts_h, sizeof(uint)));

  CUDA_CHECK(cudaMallocHost((void**)&packed_pts_out_h, sizeof(PointT) * max_pts_in_cld_));
  CUDA_CHECK(cudaMallocHost((void**)&metas_h, sizeof(PointMeta) * max_pts_in_cld_));

#ifdef VIZ
  CUDA_CHECK(cudaMallocHost((void**)&num_pts_in_patch_h, num_pts_in_patch_size));
  CUDA_CHECK(cudaMallocHost((void**)&patches_h, patches_size));
#endif // VIZ

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
  CUDA_CHECK(cudaMemsetAsync(patch_offsets_d, 0, num_pts_in_patch_size, stream));
  CUDA_CHECK(cudaMemsetAsync(cov_mats_d, 0, sizeof(float) * 3 * 3 * num_total_sectors_, stream));
  CUDA_CHECK(cudaMemsetAsync(pca_features_d, 0, num_total_sectors_ * sizeof(PCAFeature), stream));
  CUDA_CHECK(cudaMemsetAsync(eigen_vals_d, 0, num_total_sectors_ * 3 * sizeof(float), stream));
  CUDA_CHECK(cudaMemsetAsync(eig_info_d, 0, num_total_sectors_ * sizeof(int), stream));
  *num_patched_pts_h = 0;
}
template<typename PointT>
void PatchWorkGPU<PointT>::to_CUDA( pcl::PointCloud<PointT>* pc, cudaStream_t stream)
{
  CUDA_CHECK(cudaMemcpyAsync(cloud_in_d_, pc->points.data(), pc->points.size() * sizeof(PointT),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());
}


template <typename PointT>
void PatchWorkGPU<PointT>::to_pcl(pcl::PointCloud<PointT>* ground,
                                  pcl::PointCloud<PointT>* nonground)
{
  cudaMemcpyAsync(packed_pts_out_h, patches_d, sizeof(PointT) * (*num_patched_pts_h),
                  cudaMemcpyDeviceToHost, streamd2h_);
  ground->reserve(*num_patched_pts_h);
  nonground->reserve(*num_patched_pts_h);
  cudaMemcpyAsync(metas_h, metas_d, sizeof(PointMeta) * (*num_patched_pts_h),
                  cudaMemcpyDeviceToHost, streamd2h_); // dont synch for this op yet.
  // parse into ground pcl cloud
  cudaStreamSynchronize(streamd2h_); // guarantee packed_pts_out_h loaded from device
  for (size_t i=0; i<(*num_patched_pts_h); i++)
  {
    const PointT& pt = packed_pts_out_h[i];
    if (metas_h[i].ground)
    {
      ground->push_back(pt);
    } else {
      nonground->push_back(pt);
    }
  }
}

template<typename PointT>
void PatchWorkGPU<PointT>::estimate_ground(pcl::PointCloud<PointT>* cloud_in,
                                           pcl::PointCloud<PointT>* ground,
                                           pcl::PointCloud<PointT>* nonground)
{
  // TODO: sensor height estimation is not implemented yet
  reset_buffers();
  to_CUDA(cloud_in, stream_);
  bool ret = zone_model_->create_patches_gpu(cloud_in_d_, cloud_in->points.size(),
                                             num_pts_in_patch_d,  in_metas_d, metas_d,
                                             patch_offsets_d, num_total_sectors_,
                                             patches_d, *num_patched_pts_h, stream_);
  if(!ret)
  {
    throw std::runtime_error("Failed to launch create patches kernel.");
  }

  extract_init_seeds_gpu();
  fit_regionwise_planes_gpu();
//  finalize_groundness_gpu();
  cudaDeviceSynchronize();
  to_pcl(ground,nonground);
}


# ifdef VIZ
template<typename PointT>
void PatchWorkGPU<PointT>::viz_points( pcl::PointCloud<PointT>* patched_pc,
                                      pcl::PointCloud<PointT>* seed_pc)
{
  CUDA_CHECK(cudaMemcpyAsync(metas_h, metas_d, sizeof(PointMeta) * max_pts_in_cld_,
                             cudaMemcpyDeviceToHost, streamd2h_));
  CUDA_CHECK(cudaMemcpyAsync(patches_h, patches_d, patches_size,
                             cudaMemcpyDeviceToHost, streamd2h_));
  CUDA_CHECK(cudaMemcpyAsync(num_pts_in_patch_h, num_pts_in_patch_d, num_pts_in_patch_size,
                             cudaMemcpyDeviceToHost, streamd2h_));
  CUDA_CHECK(cudaMemcpyAsync(patch_offsets_h, patch_offsets_d, num_pts_in_patch_size,
                             cudaMemcpyDeviceToHost, streamd2h_));
  cudaStreamSynchronize(streamd2h_);
  uint32_t num_patched_pts {0};
  static auto& color_map = zone_model_->color_map;
  for(int ring_idx=0; ring_idx<zone_model_->num_total_rings_; ring_idx++)
  {
    auto ring_offset = std::accumulate(zone_model_->num_sectors_per_ring_.begin(),
                                       zone_model_->num_sectors_per_ring_.begin() + ring_idx , 0);
    for(int sector_idx=0; sector_idx< zone_model_->num_sectors_per_ring_[ring_idx]; sector_idx++)
    {
      auto patch_numel_offset =  ring_offset + sector_idx;
      uint num_pts = *(num_pts_in_patch_h + patch_numel_offset);

      if (num_pts >= MAX_POINTS_PER_PATCH)
      {
        throw std::runtime_error("Number of points in a patch exceeds the maximum limit.");
      }

      for(std::size_t pt_idx=0; pt_idx<num_pts; pt_idx++)
      {
        std::size_t lin_patch_offset = static_cast<std::size_t>(patch_offsets_h[patch_numel_offset]) + pt_idx;
        PointT& pt_loc = patches_h[lin_patch_offset];
        pt_loc.intensity = color_map[ring_idx * zone_model_->max_num_sectors_ + sector_idx];
        patched_pc->points.push_back(pt_loc); // Add a duplicate point with zero intensity

        if (metas_h[lin_patch_offset].ground) {
          // encode ring,sector info as intensity to be colorized when visualized
          PointT tmp_pt;
          tmp_pt.x = pt.x;
          tmp_pt.y = pt.y;
          tmp_pt.z = pt.z;
          tmp_pt.intensity = 100;
          seed_pc->points.push_back(tmp_pt);
          num_patched_pts++;
          PointT lbr_pt;
          lbr_pt.x = pt.x;
          lbr_pt.y = pt.y;
          lbr_pt.z = metas_h[lin_patch_offset].lbr;
          lbr_pt.intensity = 0; // zero intensity for the LBR point
          seed_pc->points.push_back(lbr_pt);
        }
      }
    }
  }
}
#endif

template class PatchWorkGPU<PointXYZILID>;

