//
// Created by Hyungtae Lim on 6/23/21.
// Modified to support both file input and PointCloud2 topic input
//

// For disable PCL complile lib, to use PointXYZILID
#define PCL_NO_PRECOMPILE
#include <signal.h>

#include <patchwork_cuda/node.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include "label_generator/label_generator.hpp"
#include "patchwork_gpu/patchwork_gpu.cuh"

#include "tools/kitti_loader.hpp"

using PointType = PointXYZILID;
using namespace std;

// ROS Publishers
ros::Publisher CloudPublisher;
ros::Publisher TPPublisher;
ros::Publisher FPPublisher;
ros::Publisher FNPublisher;
ros::Publisher TNPublisher;
ros::Publisher PrecisionPublisher;
ros::Publisher RecallPublisher;
ros::Publisher EstGroundPublisher;
ros::Publisher EstNonGroundPublisher;  // Publisher for non-ground points
ros::Publisher EstGroundFilteredPublisher;

// ROS Subscriber (only used in topic mode)
ros::Subscriber PointCloudSubscriber;

boost::shared_ptr<PatchWorkGPU<PointType>> PatchworkGroundSeg;

// Configuration parameters
bool use_topic_input;  // Main parameter to switch between file and topic input
std::string input_topic;
std::string data_path;
std::string algorithm;
std::string seq;
bool save_flag;
bool use_sor_before_save;
bool enable_benchmarking;
int start_frame, end_frame;

// File processing variables
std::string abs_save_dir;
std::string output_filename;
pcl::PointCloud<PointType>::Ptr filtered;

void signal_callback_handler(int signum) {
  cout << "Caught Ctrl + c " << endl;
  // Terminate program
  exit(signum);
}

void pub_score(std::string mode, double measure) {
  static int SCALE = 5;
  visualization_msgs::Marker marker;
  marker.header.frame_id = PatchworkGroundSeg->frame_patchwork;
  marker.header.stamp = ros::Time::now();
  marker.ns = "patchwork_scores";
  marker.id = (mode == "p") ? 0 : 1;
  marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker.action = visualization_msgs::Marker::ADD;
  
  if (mode == "p") marker.pose.position.x = 28.5;
  if (mode == "r") marker.pose.position.x = 25;
  marker.pose.position.y = 30;

  marker.pose.position.z = 1;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = SCALE;
  marker.scale.y = SCALE;
  marker.scale.z = SCALE;
  marker.color.a = 1.0;  // Don't forget to set the alpha!
  marker.color.r = 0.0;
  marker.color.g = 1.0;
  marker.color.b = 0.0;
  // only if using a MESH_RESOURCE marker type:
  marker.text = mode + ": " + std::to_string(measure);
  if (mode == "p") PrecisionPublisher.publish(marker);
  if (mode == "r") RecallPublisher.publish(marker);
}

template <typename T>
sensor_msgs::PointCloud2 cloud2msg(pcl::PointCloud<T> cloud, std::string frame_id, ros::Time stamp = ros::Time::now()) {
  sensor_msgs::PointCloud2 cloud_ROS;
  pcl::toROSMsg(cloud, cloud_ROS);
  cloud_ROS.header.frame_id = frame_id;
  cloud_ROS.header.stamp = stamp;
  return cloud_ROS;
}

void processPointCloud(const pcl::PointCloud<PointType>& pc_curr, 
                      const std::string& frame_id, 
                      const ros::Time& stamp,
                      int frame_number = -1) {
  
  // Ground segmentation
  pcl::PointCloud<PointType> pc_ground;
  pcl::PointCloud<PointType> pc_non_ground;
  
  static float time_taken{0.0};
  PatchworkGroundSeg->estimate_ground(const_cast<pcl::PointCloud<PointType>*>(&pc_curr), 
                                     &pc_ground, &pc_non_ground, &time_taken);
  
  if (frame_number >= 0) {
    cout << frame_number << "th: \033[1;32m"
         << " takes " << time_taken << " sec, " << pc_curr.size() << " -> " << pc_ground.size()
         << "\033[0m\n";
  } else {
    ROS_INFO("Ground segmentation completed: %zu -> %zu ground points, %zu non-ground points (%.2f ms)", 
             pc_curr.size(), pc_ground.size(), pc_non_ground.size(), time_taken);
  }
  
  // Calculate precision/recall if benchmarking is enabled and ground truth is available
  if (enable_benchmarking) {
    double precision, recall;
    try {
      calculate_precision_recall(pc_curr, pc_ground, precision, recall);
      
      if (frame_number >= 0) {
        cout << "\033[1;32mP: " << precision << " | R: " << recall << "\033[0m\n";
      } else {
        ROS_INFO("Precision: %.4f | Recall: %.4f", precision, recall);
      }
      
      // Save results to file if enabled
      if (save_flag) {
        const char *home_dir = std::getenv("HOME");
        if (home_dir != nullptr) {
          std::string output_filename;
          if (use_topic_input) {
            output_filename = std::string(home_dir) + "/patchwork_online_results.txt";
            ofstream ground_output(output_filename, ios::app);
            ground_output << stamp.toSec() << "," << time_taken << "," 
                         << precision << "," << recall << std::endl;
            ground_output.close();
          } else {
            output_filename = std::string(home_dir) + "/patchwork_quantitaive_results.txt";
            ofstream ground_output(output_filename, ios::app);
            ground_output << frame_number << "," << time_taken << "," << precision << "," << recall << std::endl;
            ground_output.close();
          }
        }
      }
      
      // Publish precision/recall markers
      pub_score("p", precision);
      pub_score("r", recall);
      
      // Create TP, FP, FN, TN clouds for visualization
      pcl::PointCloud<PointType> TP, FP, FN, TN;
      discern_ground(pc_ground, TP, FP);
      discern_ground(pc_non_ground, FN, TN);
      
      // Publish benchmark clouds
      TPPublisher.publish(cloud2msg(TP, frame_id, stamp));
      FPPublisher.publish(cloud2msg(FP, frame_id, stamp));
      FNPublisher.publish(cloud2msg(FN, frame_id, stamp));
      TNPublisher.publish(cloud2msg(TN, frame_id, stamp));
    } catch (const std::exception& e) {
      if (use_topic_input) {
        ROS_WARN("Benchmarking failed (ground truth may not be available): %s", e.what());
      } else {
        cout << "Benchmarking failed: " << e.what() << endl;
      }
    }
  }
  
  // Publish results - including non-ground points
  CloudPublisher.publish(cloud2msg(pc_curr, frame_id, stamp));
  EstGroundPublisher.publish(cloud2msg(pc_ground, frame_id, stamp));
  EstNonGroundPublisher.publish(cloud2msg(pc_non_ground, frame_id, stamp)); // Publish non-ground points
  
  // Apply statistical outlier removal if enabled
  if (use_sor_before_save) {
    filtered.reset(new pcl::PointCloud<PointType>());
    filter_by_sor(pc_ground, *filtered);
    EstGroundFilteredPublisher.publish(cloud2msg(*filtered, frame_id, stamp));
  }
  
  // Save ground labels for file processing mode
  if (!use_topic_input && save_flag && frame_number >= 0) {
    if (use_sor_before_save) {
      save_ground_label(abs_save_dir, frame_number, pc_curr, *filtered);
    } else {
      save_ground_label(abs_save_dir, frame_number, pc_curr, pc_ground);
    }
  }
}

void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
  // Convert ROS PointCloud2 to PCL point cloud
  pcl::PointCloud<PointType> pc_curr;
  
  // Handle conversion based on input point cloud type
  try {
    // Try direct conversion first (if input already has XYZILID fields)
    pcl::fromROSMsg(*msg, pc_curr);
  } catch (const std::exception& e) {
    // If direct conversion fails, convert from standard PointXYZ/PointXYZI
    sensor_msgs::PointCloud2 temp_msg = *msg;
    
    // Check if we have intensity field
    bool has_intensity = false;
    for (const auto& field : temp_msg.fields) {
      if (field.name == "intensity") {
        has_intensity = true;
        break;
      }
    }
    
    if (has_intensity) {
      // Convert from PointXYZI
      pcl::PointCloud<pcl::PointXYZI> temp_cloud;
      try {
        pcl::fromROSMsg(*msg, temp_cloud);
        
        // Convert to our custom point type
        pc_curr.resize(temp_cloud.size());
        for (size_t i = 0; i < temp_cloud.size(); ++i) {
          pc_curr[i].x = temp_cloud[i].x;
          pc_curr[i].y = temp_cloud[i].y;
          pc_curr[i].z = temp_cloud[i].z;
          pc_curr[i].intensity = temp_cloud[i].intensity;
          pc_curr[i].label = 0;    // Default label
          pc_curr[i].id = static_cast<uint16_t>(i); // Default ID
        }
      } catch (const std::exception& e2) {
        ROS_ERROR("Failed to convert PointXYZI message: %s", e2.what());
        return;
      }
    } else {
      // Convert from PointXYZ
      pcl::PointCloud<pcl::PointXYZ> temp_cloud;
      try {
        pcl::fromROSMsg(*msg, temp_cloud);
        
        // Convert to our custom point type
        pc_curr.resize(temp_cloud.size());
        for (size_t i = 0; i < temp_cloud.size(); ++i) {
          pc_curr[i].x = temp_cloud[i].x;
          pc_curr[i].y = temp_cloud[i].y;
          pc_curr[i].z = temp_cloud[i].z;
          pc_curr[i].intensity = 0.0f; // Default intensity
          pc_curr[i].label = 0;    // Default label
          pc_curr[i].id = static_cast<uint16_t>(i); // Default ID
        }
      } catch (const std::exception& e2) {
        ROS_ERROR("Failed to convert PointXYZ message: %s", e2.what());
        return;
      }
    }
  }
  
  if (pc_curr.empty()) {
    ROS_WARN("Received empty point cloud");
    return;
  }
  
  // Process the point cloud
  processPointCloud(pc_curr, msg->header.frame_id, msg->header.stamp);
}

void processFileSequence() {
  cout << "Target data: " << data_path << endl;
  KittiLoader loader(data_path);
  double p_cum{0}, r_cum{0};
  int cnt{0};
  double cum_time = 0;
  int N = loader.size();
  
  for (int n = max(0, start_frame); n < min(N, end_frame); ++n) {
    pcl::PointCloud<PointType> pc_curr;
    loader.get_cloud(n, pc_curr);
    
    // Process the point cloud
    processPointCloud(pc_curr, PatchworkGroundSeg->frame_patchwork, ros::Time::now(), n);
    
    // Accumulate statistics for file processing
    if (enable_benchmarking) {
      // Note: precision/recall are calculated inside processPointCloud
      // Here we could accumulate them if needed for final statistics
      cnt++;
    }
    
    // Small delay to allow visualization
    ros::Duration(0.1).sleep();
    ros::spinOnce();
  }
  
  if (cnt > 0) {
    std::cout << "\033[1;34mProcessed " << cnt << " frames\033[0m" << std::endl;
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "HybridPatchWorkGPU");
  
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");
  
  // Get main parameter to determine input mode
  private_nh.param<bool>("use_topic_input", use_topic_input, false);
  
  // Common parameters
  private_nh.param<bool>("save_flag", save_flag, false);
  private_nh.param<bool>("use_sor_before_save", use_sor_before_save, false);
  private_nh.param<bool>("enable_benchmarking", enable_benchmarking, false);
  private_nh.param<std::string>("algorithm", algorithm, "patchwork");
  
  if (use_topic_input) {
    // Topic-based parameters
    private_nh.param<std::string>("input_topic", input_topic, "/lslidar_point_cloud");
    
    ROS_INFO("=== TOPIC INPUT MODE ===");
    ROS_INFO("Subscribing to point cloud topic: %s", input_topic.c_str());
  } else {
    // File-based parameters
    private_nh.param<std::string>("data_path", data_path, "/");
    private_nh.param<std::string>("seq", seq, "00");
    private_nh.param<int>("start_frame", start_frame, 0);
    private_nh.param<int>("end_frame", end_frame, 10000);
    
    ROS_INFO("=== FILE INPUT MODE ===");
    ROS_INFO("Processing files from: %s", data_path.c_str());
    ROS_INFO("Frame range: %d to %d", start_frame, end_frame);
    
    // Setup save directory for file mode
    if (save_flag) {
      abs_save_dir = data_path + "/patchwork";
      std::cout << "\033[1;34m" << abs_save_dir << "\033[0m" << std::endl;
      std::experimental::filesystem::create_directory(abs_save_dir);
    }
  }
  
  ROS_INFO("Save flag: %s", save_flag ? "true" : "false");
  ROS_INFO("Use SOR before save: %s", use_sor_before_save ? "true" : "false");
  ROS_INFO("Enable benchmarking: %s", enable_benchmarking ? "true" : "false");
  
  // Initialize publishers
  CloudPublisher = nh.advertise<sensor_msgs::PointCloud2>("/patchwork/cloud", 100, true);
  EstGroundPublisher = nh.advertise<sensor_msgs::PointCloud2>("/patchwork/ground", 100, true);
  EstNonGroundPublisher = nh.advertise<sensor_msgs::PointCloud2>("/patchwork/non_ground", 100, true);
  EstGroundFilteredPublisher = nh.advertise<sensor_msgs::PointCloud2>("/patchwork/ground_filtered", 100, true);
  
  // Benchmarking publishers (only if benchmarking is enabled)
  if (enable_benchmarking) {
    TPPublisher = nh.advertise<sensor_msgs::PointCloud2>("/benchmark/TP", 100, true);
    FPPublisher = nh.advertise<sensor_msgs::PointCloud2>("/benchmark/FP", 100, true);
    FNPublisher = nh.advertise<sensor_msgs::PointCloud2>("/benchmark/FN", 100, true);
    TNPublisher = nh.advertise<sensor_msgs::PointCloud2>("/benchmark/TN", 100, true);
    PrecisionPublisher = nh.advertise<visualization_msgs::Marker>("/patchwork/precision", 1, true);
    RecallPublisher = nh.advertise<visualization_msgs::Marker>("/patchwork/recall", 1, true);
  }
  
  // Initialize PatchWork
  PatchworkGroundSeg.reset(new PatchWorkGPU<PointType>(&nh));
  
  // Set up signal handler
  signal(SIGINT, signal_callback_handler);
  
  if (use_topic_input) {
    // Topic-based processing
    PointCloudSubscriber = nh.subscribe(input_topic, 1, pointCloudCallback);
    ROS_INFO("PatchWork GPU node started. Waiting for point cloud messages...");
    ros::spin();
  } else {
    // File-based processing
    ROS_INFO("PatchWork GPU node started. Processing file sequence...");
    processFileSequence();
    ROS_INFO("File processing completed.");
  }
  
  return 0;
}