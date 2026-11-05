#pragma once

#include "common/namespace.h"
#include "common/basic_type.h"
// #include "scanner/trajectory.h"
NMRECON_NAMESPACE_BEGIN
namespace scanner
{
    typedef nmrecon::common::Matrix4x4 Affine;
    typedef thrust::device_vector<Affine> AffineList;

    /// @brief
    struct TrajectoryView
    {
        precision_t view_angle;
        precision_t radius;
        __host__ __device__ TrajectoryView(const precision_t &view_angle_ = 0, const precision_t &radius_ = 10)
            : view_angle(view_angle_), radius(radius_) {};
    };

    struct affine_view_functor
    {
        affine_view_functor(){};
        __device__ __host__ Affine operator()(const TrajectoryView &view) const;
    };

    struct affine_functor
    {
        const precision_t radius;
        const precision_t start_angle;
        const precision_t end_angle;
        const size_t num_views;

        affine_functor(const precision_t &radius_, const precision_t &start_angle_, const precision_t &end_angle_, const size_t &num_views_)
            : radius(radius_), start_angle(start_angle_), end_angle(end_angle_), num_views(num_views_) {};
        __device__ __host__ Affine operator()(const size_t &index) const;
    };

}
NMRECON_NAMESPACE_END