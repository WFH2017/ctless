#include <iostream>
#include "scanner/trajectory_view.h"

NMRECON_NAMESPACE_BEGIN
namespace scanner
{
    __device__ __host__ Affine affine_view_functor::operator()(const TrajectoryView &trajectory_view) const
    {
        auto aff_translation = Affine::make_translation(trajectory_view.radius, 0, 0);
        auto aff_rotation = Affine::make_rotation_z(trajectory_view.view_angle);
        auto aff = aff_rotation.compose(aff_translation);
        return aff;
    }
    __device__ __host__ Affine affine_functor::operator()(const size_t &index) const
    {
        precision_t theta = start_angle;
        if (num_views > 1)
        {
            theta = start_angle + (end_angle - start_angle) * index / num_views;
        }
        auto aff_translation = Affine::make_translation(radius, 0, 0);
        auto aff_rotation = Affine::make_rotation_z(theta);
        auto aff = aff_rotation.compose(aff_translation);
        return aff;
    }
}
NMRECON_NAMESPACE_END