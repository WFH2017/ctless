#pragma once
#include <stdio.h>
#include <vector>
#include "common/config.h"
#include "common/basic_type.h"
#include "scanner/trajectory_view.h"

NMRECON_NAMESPACE_BEGIN

namespace scanner
{

  struct CircularTrajectory
  {
    precision_t radius;
    precision_t start_angle;
    precision_t end_angle;
    size_t num_views;

    CircularTrajectory(const precision_t &radius_ = 10, const precision_t &start_angle_ = 0, const precision_t &end_angle_ = 180, const size_t &num_views_ = 2)
        : radius(radius_), start_angle(start_angle_), end_angle(end_angle_), num_views(num_views_) {};

    CircularTrajectory(const CircularTrajectory &other)
        : radius(other.radius), start_angle(other.start_angle), end_angle(other.end_angle), num_views(other.num_views) {};

    precision_t get_step_angle() const
    {
      if (num_views == 1)
        return 0.0;
      return (end_angle - start_angle) / (num_views - 1);
    }

    precision_t get_view_angle(const size_t index) const
    {
      return start_angle + get_step_angle() * index;
    }

    size_t get_num_views() const
    {
      return num_views;
    }

    thrust::host_vector<CircularTrajectory> get_sub_trajectories(const size_t num_sub_trajectories) const;

    AffineList get_affine_list() const;

    void print() const;
  };

  struct AixalTrajectory
  {
    precision_t start_z;
    precision_t end_z;
    size_t num_views;
  };

  struct BodyContourTrajectory
  {
    std::vector<precision_t> radius_vector;
    precision_t start_angle;
    precision_t end_angle;
    size_t num_views;

     // for backward compatibility with single radius input
    BodyContourTrajectory(const precision_t& radius_= 10, const precision_t &start_angle_= 0, const precision_t &end_angle_= 180,
                          const size_t &num_views_= 2) : radius_vector(std::vector<precision_t>(num_views_, radius_)), start_angle(start_angle_), end_angle(end_angle_), num_views(num_views_) {};
    BodyContourTrajectory(const std::vector<precision_t> &radius_vector_, const precision_t &start_angle_, const precision_t &end_angle_,
                          const size_t &num_views_) : radius_vector(radius_vector_), start_angle(start_angle_), end_angle(end_angle_), num_views(num_views_) {};
   
    BodyContourTrajectory(const BodyContourTrajectory &other)
        : radius_vector(other.radius_vector), start_angle(other.start_angle), end_angle(other.end_angle), num_views(other.num_views) {};

    precision_t get_step_angle() const
    {
      if (num_views == 1)
        return 0.0;
      return (end_angle - start_angle) / num_views;
    }

    precision_t get_view_angle(const size_t index) const
    {
      return start_angle + get_step_angle() * index;
    }

    size_t get_num_views() const
    {
      return num_views;
    }

    thrust::device_vector<TrajectoryView> get_trajectory_views() const;
    
    thrust::host_vector<BodyContourTrajectory> get_sub_trajectories(const size_t num_sub_trajectories) const;
    AffineList get_affine_list() const;
    void print() const;
  };

} // end namespace scanner

NMRECON_NAMESPACE_END
