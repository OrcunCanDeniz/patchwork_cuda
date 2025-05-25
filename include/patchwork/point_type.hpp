//
// Created by orcun on 25.05.2025.
//

#ifndef PATCHWORK_POINT_TYPE_HPP
#define PATCHWORK_POINT_TYPE_HPP

#include <pcl/point_types.h>

struct PointXYZILID {
  PCL_ADD_POINT4D;  // quad-word XYZ
  float intensity;  ///< laser intensity reading
  uint16_t label;   ///< point label
  uint16_t id;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW  // ensure proper alignment
} EIGEN_ALIGN16;
#endif  // PATCHWORK_POINT_TYPE_HPP
