//
// Created by orcun on 10.05.2025.
//

#include "patchwork_gpu.cuh"


template<typename PointT>
void PatchWorkGPU<PointT>::toCUDA(const pcl::PointCloud<PointT> &cloud_in) {
  CUDA_CHECK(cudaMemcpyAsync(cloud_in_d, cloud_in.points.data(), cloud_in.points.size(),
                  cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());
}




