//
// Created by Hyungtae Lim on 6/23/21.
//

// For disable PCL complile lib, to use PointXYZILID
#define PCL_NO_PRECOMPILE
#include <signal.h>

#include <pcl/common/centroid.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include "patchwork_gpu/patchwork_gpu.cuh"
#include "tools/kitti_loader.hpp"

using PointType = PointXYZILID;
using namespace std;

ros::Publisher CloudPublisher;

boost::shared_ptr<PatchWorkGPU<PointType>> PatchworkGroundSegGPU;

std::string abs_save_dir;
std::string output_filename;
std::string acc_filename;
std::string pcd_savepath;
std::string data_path;
std::string algorithm;
std::string seq;
bool save_flag;
bool use_sor_before_save;

pcl::PointCloud<PointType>::Ptr filtered;

void signal_callback_handler(int signum) {
  cout << "Caught Ctrl + c " << endl;
  // Terminate program
  exit(signum);
}


template <typename T>
sensor_msgs::PointCloud2 cloud2msg(pcl::PointCloud<T> cloud, std::string frame_id) {
  sensor_msgs::PointCloud2 cloud_ROS;
  pcl::toROSMsg(cloud, cloud_ROS);
  cloud_ROS.header.frame_id = frame_id;
  return cloud_ROS;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "OfflineKITTI");

  ros::NodeHandle nh;
  int start_frame, end_frame;
  condParam<int>(&nh, "/start_frame", start_frame, 0, "");
  condParam<int>(&nh, "/end_frame", end_frame, 10000, "");
  condParam<bool>(&nh, "/save_flag", save_flag, false, "");
  condParam<bool>(&nh, "/use_sor_before_save", use_sor_before_save, false, "");
  condParam<string>(&nh, "/algorithm", algorithm, "patchwork", "");
  condParam<string>(&nh, "/seq", seq, "00", "");
  condParam<string>(&nh, "/data_path", data_path,
                    "/media/orcun/orcun2tb2/kitti/labeled_dataset/sequences/04", "");

  CloudPublisher = nh.advertise<sensor_msgs::PointCloud2>("/benchmark/og_cloud", 100, true);

#ifdef VIZ_PATCHES
  ROS_WARN("Visualizing patches is enabled. This may cause performance issues.");
  ros::Publisher PatchedPublisher = nh.advertise<sensor_msgs::PointCloud2>("/benchmark/patches", 100, true);
#endif


  signal(SIGINT, signal_callback_handler);

  float sensor_height_, max_range_, min_range_;
  std::string sensor_model_;
  condParam(&nh, "sensor_model", sensor_model_, std::string("HDL-64E"), "");
  condParam<float>(&nh, "sensor_height", sensor_height_, 1.723, "");
  condParam<float>(&nh, "max_r", max_range_, 80.0);
  condParam<float>(&nh, "min_r", min_range_, 2.7);

  PatchworkGroundSegGPU.reset(new PatchWorkGPU<PointType>(&nh));

  cout << "Target data: " << data_path << endl;
  KittiLoader loader(data_path);

  int N = loader.size();
  for (int n = max(0, start_frame); n < min(N, end_frame); ++n) {
    pcl::PointCloud<PointType> pc_curr;
    pcl::PointCloud<PointType> patch_pc;

    loader.get_cloud(n, pc_curr);
    CloudPublisher.publish(cloud2msg(pc_curr, "map"));

    PatchworkGroundSegGPU->zone_model_->reset_buffers();
    PatchworkGroundSegGPU->estimate_ground(&pc_curr);

#ifdef VIZ_PATCHES
    patch_pc.reserve(pc_curr.size());
    uint32_t np = PatchworkGroundSegGPU->get_patched_cloud(&patch_pc);//this causes crash
    PatchedPublisher.publish(cloud2msg(patch_pc, "map"));
    patch_pc.clear();
#endif

    std::cout<<"frame idx: "<<n<<std::endl;

    while (std::cin.get() != ' ') {
      // Wait for space bar input
    }
  }
  return 0;
}
