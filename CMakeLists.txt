cmake_minimum_required(VERSION 3.0.2)
project(patchwork)


option(VIZ "VIZ" OFF)

if (VIZ)
        add_compile_definitions(VIZ)
endif (VIZ)

add_compile_options(-std=c++17)
set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES "Auto")
list(APPEND CUDA_NVCC_FLAGS
        -Xcudafe "--diag_suppress=20236"
        -Xcudafe "--diag_suppress=20012"
        -Xcudafe "--diag_suppress=20014"
        -Xcudafe "--diag_suppress=20015"
        -Xcudafe "--diag_suppress=20013"
        -Xcompiler -fPIC
        -rdc=true
)


set(CMAKE_CUDA_STANDARD 14)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_RESOLVE_DEVICE_SYMBOLS ON)

find_package(catkin REQUIRED COMPONENTS
        geometry_msgs
        message_generation
        pcl_ros
        roscpp
        rospy
        std_msgs
        sensor_msgs
        )

add_message_files(
        FILES
        node.msg
        ground_estimate.msg
)

generate_messages(
        DEPENDENCIES
        std_msgs
        geometry_msgs
        sensor_msgs
)

find_package(PCL 1.7 REQUIRED)
find_package(Boost 1.54 REQUIRED)

catkin_package(
        INCLUDE_DIRS
        LIBRARIES
        CATKIN_DEPENDS geometry_msgs message_runtime roscpp rospy sensor_msgs std_msgs
)

option(USE_SYSTEM_TBB "Use system pre-installed oneAPI/tbb" OFF)
include(3rdparty/find_dependencies.cmake)

include_directories(
        ${catkin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        include
)

cuda_add_library(
        zone_models_gpu
        lib/zone_models_kernels.cu
        OPTIONS ${CUDA_NVCC_FLAGS}
)

cuda_add_library(patchwork_gpu lib/patchwork_kernels.cu)

cuda_add_library(patchwork_host lib/patchwork_host.cpp)
target_link_libraries(patchwork_host
        ${catkin_LIBRARIES}
        ${CUDA_cusolver_LIBRARY}
        ${PCL_LIBRARIES}
        zone_models_gpu
        patchwork_gpu
)

add_executable(patchwork_workbench nodes/patchwork_workbench.cpp)
target_link_libraries(patchwork_workbench
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        zone_models_gpu
        patchwork_host)

add_executable(ros_kitti nodes/ros_kitti.cpp)
target_link_libraries(ros_kitti ${catkin_LIBRARIES} ${PCL_LIBRARIES} TBB::tbb)
add_dependencies(ros_kitti patchwork_generate_messages_cpp)

add_executable(ros_kitti_publisher nodes/ros_kitti_publisher.cpp)
target_link_libraries(ros_kitti_publisher ${catkin_LIBRARIES} ${PCL_LIBRARIES} TBB::tbb)
add_dependencies(ros_kitti_publisher patchwork_generate_messages_cpp)

add_executable(pub_for_legoloam nodes/pub_for_legoloam.cpp)
target_link_libraries(pub_for_legoloam ${catkin_LIBRARIES} ${PCL_LIBRARIES} TBB::tbb)
add_dependencies(pub_for_legoloam patchwork_generate_messages_cpp)

add_executable(offline_kitti nodes/offline_kitti.cpp
        include/patchwork_gpu/patchwork_gpu.cuh)
target_link_libraries(offline_kitti ${catkin_LIBRARIES} ${PCL_LIBRARIES} stdc++fs TBB::tbb)
add_dependencies(offline_kitti patchwork_generate_messages_cpp)

add_executable(offline_own_data nodes/offline_own_data.cpp)
target_link_libraries(offline_own_data ${catkin_LIBRARIES} ${PCL_LIBRARIES} TBB::tbb)
add_dependencies(offline_own_data patchwork_generate_messages_cpp)
