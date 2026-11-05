#include "scanner/collimator.h"

NMRECON_NAMESPACE_BEGIN
namespace scanner
{
    __device__ Vec3 hexagon_lookup_table_functor::operator()(const size_t &index) const
    {
        const size_t ind_x = index % size.x;
        const size_t ind_y = (index / size.x) % size.y;
        const size_t ind_z = index / (size.x * size.y);

        /**
         * the offset of even columns on the collimator for hexagonal arrangement.
         * if the ind_z is odd (even columns), the offset is zero.
         * */
        // Point ind_T(ind_x, ind_y, ind_z);
        // Point even_origin_offset(0,0,spacing.z/2*(ind_y%2));
        // auto transformd_spacing = direction.transform(spacing);
        // return origin + even_origin_offset + ind_T * transformd_spacing;

        Vec3 ind_T(ind_x, ind_y, ind_z);
        auto transformd_spacing = direction.transform(spacing);
        return origin + ind_T * transformd_spacing;
    }

    __host__ Point3DVector ParallelCollimator::make_lookup_table() const
    {
      size_t num_voxels = get_num_rays();
      thrust::counting_iterator<size_t> index_sequence_begin(0);
      Point3DVector lookup_table(num_voxels);
      thrust::transform(thrust::device, index_sequence_begin, index_sequence_begin + num_voxels, lookup_table.begin(),
                        hexagon_lookup_table_functor(coordinate_prototype.origin, coordinate_prototype.size,
                                                     coordinate_prototype.spacing, coordinate_prototype.direction));
      return lookup_table;
    }
}

NMRECON_NAMESPACE_END
