#include "cuda/reconstruction/distance_driven.h"

NMRECON_NAMESPACE_BEGIN
namespace reconstruction
{
    namespace projector
    {
        namespace distance_driven
        {
            const int kThreadPerBlock_x = 512;
            const int kThreadPerBlock_y = 1;
            // const precision_t kEpisilon = 1e-6f;
            // const precision_t kMinRatio = 0.001f;
            // const precision_t kMaxRatio = 0.999f;

            /**
             * @brief for projection without attenuation
             *
             * @tparam T
             * @param image
             * @param bound_min
             * @param bound_max
             * @param pos1_det
             * @param pos2_det
             * @return __device__
             */
            __device__ T loop_yz_plane(const DeviceImage3D &image,
                                       const IndexInt &bound_min, const IndexInt &bound_max,
                                       const Value3<T> &pos1_det, const Value3<T> &pos2_det)
            {
                T value = 0;
                T weight_y, weight_z;
                const Value3<T> half_spacing = image.coordinate.spacing / 2;
                const Size image_size = image.coordinate.size;
                // loop over y axis
                for (int iy = max(0, bound_min.y); iy < min(int(image_size.y), bound_max.y + 1); iy++)
                {
                    // overlap of voxel mapping on y
                    weight_y = min(pos2_det.y, image.get_pos_y_by_index(iy) + half_spacing.y) - max(pos1_det.y, image.get_pos_y_by_index(iy) - half_spacing.y);
                    // divide the detector mapping length on y
                    weight_y = weight_y < 0 ? 0 : weight_y / (pos2_det.y - pos1_det.y);
                    // loop over z axis
                    for (int iz = max(0, bound_min.z); iz < min(int(image_size.z), bound_max.z + 1); iz++)
                    {
                        weight_z = min(pos2_det.z, image.get_pos_z_by_index(iz) + half_spacing.z) - max(pos1_det.z, image.get_pos_z_by_index(iz) - half_spacing.z);
                        weight_z = weight_z < 0 ? 0 : weight_z / (pos2_det.z - pos1_det.z);
                        value += weight_y * weight_z * image.get_value_by_index(bound_min.x, iy, iz);
                    }
                }
                return value;
            }

            __device__ T loop_yz_plane(const DeviceImage3D &image,
                                       const DeviceImage3D &atten_map, T &atten_integral,
                                       const T &interaction_length,
                                       const IndexInt &bound_min, const IndexInt &bound_max,
                                       const Value3<T> &pos1_det, const Value3<T> &pos2_det)
            {
                T value = 0;
                T weight_y, weight_z;
                const Value3<T> half_spacing = image.coordinate.spacing / 2;
                const Size image_size = image.coordinate.size;

                // use the accumulated value until last plane for acceleration.
                T atten_factor = __expf(-atten_integral);
                T plane_atten_integral = 0;
                T plane_weight = 0;

                // loop over y axis
                for (int iy = max(0, bound_min.y); iy < min(int(image_size.y), bound_max.y + 1); iy++)
                {
                    // overlap of voxel mapping on y
                    weight_y = min(pos2_det.y, image.get_pos_y_by_index(iy) + half_spacing.y) - max(pos1_det.y, image.get_pos_y_by_index(iy) - half_spacing.y);
                    // divide the detector mapping length on y
                    weight_y = weight_y < 0 ? 0 : weight_y / (pos2_det.y - pos1_det.y);
                    // loop over z axis
                    for (int iz = max(0, bound_min.z); iz < min(int(image_size.z), bound_max.z + 1); iz++)
                    {
                        weight_z = min(pos2_det.z, image.get_pos_z_by_index(iz) + half_spacing.z) - max(pos1_det.z, image.get_pos_z_by_index(iz) - half_spacing.z);
                        weight_z = weight_z < 0 ? 0 : weight_z / (pos2_det.z - pos1_det.z);

                        T geo_factor = weight_y * weight_z * interaction_length;
                        value += image.get_value_by_index(bound_min.x, iy, iz) * geo_factor;
                        // intergrate the attenuation map for next plane.
                        plane_atten_integral += atten_map.get_value_by_index(bound_min.x, iy, iz) * geo_factor;
                        plane_weight += geo_factor;
                    }
                }
                T atten_increment = 0;
                if (plane_weight > 0)
                    atten_increment = plane_atten_integral / plane_weight;
                atten_integral += interaction_length * atten_increment;
                return value * atten_factor;
            }

            __device__ void loop_yz_plane_bp(const T &value, DeviceImage3D &image,
                                             const IndexInt &bound_min, const IndexInt &bound_max,
                                             const Value3<T> &pos1_det, const Value3<T> &pos2_det)
            {
                T weight_y, weight_z;
                const Value3<T> half_spacing = image.coordinate.spacing / 2;
                const Size image_size = image.coordinate.size;
                // loop over y axis
                for (int iy = max(0, bound_min.y); iy < min(int(image_size.y), bound_max.y + 1); iy++)
                {
                    // overlap of voxel mapping on y
                    weight_y = min(pos2_det.y, image.get_pos_y_by_index(iy) + half_spacing.y) - max(pos1_det.y, image.get_pos_y_by_index(iy) - half_spacing.y);
                    // divide the detector mapping length on y
                    weight_y = weight_y < 0 ? 0 : weight_y / (pos2_det.y - pos1_det.y);
                    // loop over z axis
                    for (int iz = max(0, bound_min.z); iz < min(int(image_size.z), bound_max.z + 1); iz++)
                    {
                        weight_z = min(pos2_det.z, image.get_pos_z_by_index(iz) + half_spacing.z) - max(pos1_det.z, image.get_pos_z_by_index(iz) - half_spacing.z);
                        weight_z = weight_z < 0 ? 0 : weight_z / (pos2_det.z - pos1_det.z);
                        image.set_value_by_index(bound_min.x, iy, iz, value * weight_y * weight_z);
                    }
                }
            }

            __device__ void loop_yz_plane_bp(const T &value, DeviceImage3D &image,
                                             const DeviceImage3D &atten_map, T &atten_integral,
                                             const T &interaction_length,
                                             const IndexInt &bound_min, const IndexInt &bound_max,
                                             const Value3<T> &pos1_det, const Value3<T> &pos2_det)
            {

                T weight_y, weight_z;
                const Value3<T> half_spacing = image.coordinate.spacing / 2;
                const Size image_size = image.coordinate.size;
                // calculate the attenuation integral use the last accumulated value for less computation.
                T atten_factor = __expf(-atten_integral);

                T plane_atten_integral = 0;
                T plane_weight = 0;

                // loop over y axis
                for (int iy = max(0, bound_min.y); iy < min(int(image_size.y), bound_max.y + 1); iy++)
                {
                    // overlap of voxel mapping on y
                    weight_y = min(pos2_det.y, image.get_pos_y_by_index(iy) + half_spacing.y) - max(pos1_det.y, image.get_pos_y_by_index(iy) - half_spacing.y);
                    // divide the detector mapping length on y
                    weight_y = weight_y < 0 ? 0 : weight_y / (pos2_det.y - pos1_det.y);
                    // loop over z axis
                    for (int iz = max(0, bound_min.z); iz < min(int(image_size.z), bound_max.z + 1); iz++)
                    {
                        weight_z = min(pos2_det.z, image.get_pos_z_by_index(iz) + half_spacing.z) - max(pos1_det.z, image.get_pos_z_by_index(iz) - half_spacing.z);
                        weight_z = weight_z < 0 ? 0 : weight_z / (pos2_det.z - pos1_det.z);

                        T geo_factor = weight_y * weight_z * interaction_length;

                        image.set_value_by_index(bound_min.x, iy, iz, value * geo_factor * atten_factor);

                        // integrate the attenuation map for the next plane.
                        plane_atten_integral += atten_map.get_value_by_index(bound_min.x, iy, iz) * geo_factor;
                        plane_weight += geo_factor;
                    }
                }
                T atten_increment = 0;
                if (plane_weight > 0)
                    atten_increment = plane_atten_integral / plane_weight; // normalize the attenuation integral by the weight to get the average attenuation value.
                atten_integral += interaction_length * atten_increment;

                /* this loop is omitted by using the last accumulated value for acceleration.
                 T atten_factor = __expf(-atten_integral);
                for (int iy = max(0, bound_min.y); iy < min(int(image_size.y), bound_max.y+1); iy++)
                {
                  // overlap of voxel mapping on y
                  weight_y = min(pos2_det.y, image.get_pos_y_by_index(iy)+half_spacing.y) - max(pos1_det.y, image.get_pos_y_by_index(iy)-half_spacing.y);
                  // divide the detector mapping length on y
                  weight_y = weight_y < 0 ? 0 : weight_y/(pos2_det.y - pos1_det.y);
                  // loop over z axis
                  for (int iz = max(0, bound_min.z); iz < min(int(image_size.z), bound_max.z+1); iz++)
                  {
                    weight_z = min(pos2_det.z, image.get_pos_z_by_index(iz)+half_spacing.z) - max(pos1_det.z, image.get_pos_z_by_index(iz) - half_spacing.z);
                    weight_z = weight_z<0 ? 0 : weight_z/(pos2_det.z - pos1_det.z);

                    image.set_value_by_index(bound_min.x, iy, iz, value*weight_y*weight_z*atten_factor);
                  }
                }
                */
            }

            __device__ T loop_xz_plane(const DeviceImage3D &image,
                                       const IndexInt &bound_min, const IndexInt &bound_max,
                                       const Value3<T> &pos1_det, const Value3<T> &pos2_det)
            {
                T value = 0;
                T weight_x, weight_z;
                const Value3<T> half_spacing = image.coordinate.spacing / 2;
                const Size image_size = image.coordinate.size;
                // loop over x axis
                for (int ix = max(0, bound_min.x); ix < min(int(image_size.x), bound_max.x + 1); ix++)
                {
                    weight_x = min(pos2_det.x, image.get_pos_x_by_index(ix) + half_spacing.x) - max(pos1_det.x, image.get_pos_x_by_index(ix) - half_spacing.x);
                    weight_x = weight_x < 0 ? 0 : weight_x / (pos2_det.x - pos1_det.x);
                    // loop over z axis
                    for (int iz = max(0, bound_min.z); iz < min(int(image_size.z), bound_max.z + 1); iz++)
                    {
                        weight_z = min(pos2_det.z, image.get_pos_z_by_index(iz) + half_spacing.z) - max(pos1_det.z, image.get_pos_z_by_index(iz) - half_spacing.z);
                        weight_z = weight_z < 0 ? 0 : weight_z / (pos2_det.z - pos1_det.z);
                        value += weight_x * weight_z * image.get_value_by_index(ix, bound_min.y, iz);
                    }
                }
                return value;
            }

            __device__ T loop_xz_plane(const DeviceImage3D &image,
                                       const DeviceImage3D &atten_map, T &atten_integral,
                                       const T &interaction_length,
                                       const IndexInt &bound_min, const IndexInt &bound_max,
                                       const Value3<T> &pos1_det, const Value3<T> &pos2_det)
            {
                T value = 0;
                T weight_x, weight_z;
                const Value3<T> half_spacing = image.coordinate.spacing / 2;
                const Size image_size = image.coordinate.size;
                // use the accumulated value until last plane for acceleration.
                T atten_factor = __expf(-atten_integral);

                T plane_atten_integral = 0;
                T plane_weight = 0;

                // loop over x axis
                for (int ix = max(0, bound_min.x); ix < min(int(image_size.x), bound_max.x + 1); ix++)
                {
                    weight_x = min(pos2_det.x, image.get_pos_x_by_index(ix) + half_spacing.x) - max(pos1_det.x, image.get_pos_x_by_index(ix) - half_spacing.x);
                    weight_x = weight_x < 0 ? 0 : weight_x / (pos2_det.x - pos1_det.x);
                    // loop over z axis
                    for (int iz = max(0, bound_min.z); iz < min(int(image_size.z), bound_max.z + 1); iz++)
                    {
                        weight_z = min(pos2_det.z, image.get_pos_z_by_index(iz) + half_spacing.z) - max(pos1_det.z, image.get_pos_z_by_index(iz) - half_spacing.z);
                        weight_z = weight_z < 0 ? 0 : weight_z / (pos2_det.z - pos1_det.z);

                        T geo_factor = weight_x * weight_z * interaction_length;
                        value += image.get_value_by_index(ix, bound_min.y, iz) * geo_factor;
                        // intergrate the attenuation map for next plane.
                        plane_atten_integral += atten_map.get_value_by_index(ix, bound_min.y, iz) * geo_factor;
                        plane_weight += geo_factor;
                    }
                }
                T atten_increment = 0;
                if (plane_weight > 0)
                    atten_increment = plane_atten_integral / plane_weight;
                atten_integral += interaction_length * atten_increment;
                return value * atten_factor;
            }
            __device__ void loop_xz_plane_bp(const T &value, DeviceImage3D &image,
                                             const IndexInt &bound_min, const IndexInt &bound_max,
                                             const Value3<T> &pos1_det, const Value3<T> &pos2_det)
            {
                T weight_x, weight_z;
                const Value3<T> half_spacing = image.coordinate.spacing / 2;
                const Size image_size = image.coordinate.size;
                // loop over x axis
                for (int ix = max(0, bound_min.x); ix < min(int(image_size.x), bound_max.x + 1); ix++)
                {
                    weight_x = min(pos2_det.x, image.get_pos_x_by_index(ix) + half_spacing.x) - max(pos1_det.x, image.get_pos_x_by_index(ix) - half_spacing.x);
                    weight_x = weight_x < 0 ? 0 : weight_x / (pos2_det.x - pos1_det.x);
                    // loop over z axis
                    for (int iz = max(0, bound_min.z); iz < min(int(image_size.z), bound_max.z + 1); iz++)
                    {
                        weight_z = min(pos2_det.z, image.get_pos_z_by_index(iz) + half_spacing.z) - max(pos1_det.z, image.get_pos_z_by_index(iz) - half_spacing.z);
                        weight_z = weight_z < 0 ? 0 : weight_z / (pos2_det.z - pos1_det.z);
                        image.set_value_by_index(ix, bound_min.y, iz, value * weight_x * weight_z);
                    }
                }
            }

            __device__ void loop_xz_plane_bp(const T &value, DeviceImage3D &image,
                                             const DeviceImage3D &atten_map, T &atten_integral,
                                             const T &interaction_length,
                                             const IndexInt &bound_min, const IndexInt &bound_max,
                                             const Value3<T> &pos1_det, const Value3<T> &pos2_det)
            {
                T weight_x, weight_z;
                const Value3<T> half_spacing = image.coordinate.spacing / 2;
                const Size image_size = image.coordinate.size;

                // use the accumulated value until last plane for acceleration.
                T atten_factor = __expf(-atten_integral);
                // attenuation integral and sum weight for the current plane.
                T plane_atten_integral = 0;
                T plane_weight = 0;
                // loop over x axis
                for (int ix = max(0, bound_min.x); ix < min(int(image_size.x), bound_max.x + 1); ix++)
                {
                    weight_x = min(pos2_det.x, image.get_pos_x_by_index(ix) + half_spacing.x) - max(pos1_det.x, image.get_pos_x_by_index(ix) - half_spacing.x);
                    weight_x = weight_x < 0 ? 0 : weight_x / (pos2_det.x - pos1_det.x);
                    // loop over z axis
                    for (int iz = max(0, bound_min.z); iz < min(int(image_size.z), bound_max.z + 1); iz++)
                    {
                        weight_z = min(pos2_det.z, image.get_pos_z_by_index(iz) + half_spacing.z) - max(pos1_det.z, image.get_pos_z_by_index(iz) - half_spacing.z);
                        weight_z = weight_z < 0 ? 0 : weight_z / (pos2_det.z - pos1_det.z);

                        // calculate the distance-driven geometric factor g_ij
                        T geo_factor = weight_x * weight_z * interaction_length;
                        image.set_value_by_index(ix, bound_min.y, iz, value * atten_factor * geo_factor);
                        // intergrate the attenuation map for next plane, the geo_factor is used to normalize the attenuation value in current plane.
                        plane_atten_integral += atten_map.get_value_by_index(ix, bound_min.y, iz) * geo_factor;
                        plane_weight += geo_factor;
                    }
                }
                T atten_increment = 0;
                if (plane_weight > 0)
                    atten_increment = plane_atten_integral / plane_weight;
                atten_integral += interaction_length * atten_increment;
                /*this loop is omitted by using the last accumulated value for acceleration.
                T atten_factor = __expf(-atten_integral);
                for (int ix = max(0, bound_min.x); ix < min(int(image_size.x), bound_max.x+1); ix++)
                {
                  weight_x = min(pos2_det.x, image.get_pos_x_by_index(ix)+half_spacing.x) - max(pos1_det.x, image.get_pos_x_by_index(ix) - half_spacing.x);
                  weight_x = weight_x<0 ? 0 : weight_x/(pos2_det.x - pos1_det.x);
                  // loop over z axis
                  for (int iz = max(0, bound_min.z); iz < min(int(image_size.z), bound_max.z+1); iz++)
                  {
                    weight_z = min(pos2_det.z, image.get_pos_z_by_index(iz) + half_spacing.z) - max(pos1_det.z, image.get_pos_z_by_index(iz) - half_spacing.z);
                    weight_z = weight_z<0 ? 0 : weight_z/(pos2_det.z - pos1_det.z);

                    image.set_value_by_index(ix, bound_min.y, iz,  value*atten_factor*weight_x*weight_z);
                  }
                }
                */
            }

            __device__ void set_detector_bounds(const Matrix4x4 &det_affine,
                                                const Coordinate3D &det_coordinate,
                                                Value3<T> &det_left, Value3<T> &det_right,
                                                Value3<T> &det_bottom, Value3<T> &det_top)
            {
                auto det_half_spacing = det_coordinate.spacing / 2;

                // calculate the edge of two ends of lor on y axis
                const Value3<T> offset_y1(0.0f, -det_half_spacing.y, 0.0f), offset_y2(0.0f, det_half_spacing.y, 0.0f);
                det_left = det_affine.transform(offset_y1);
                det_right = det_affine.transform(offset_y2);

                // calculate the edge of two ends of lor on z axis
                const Value3<T> offset_z1(0.0f, 0.0f, -det_half_spacing.z), offset_z2(0.0f, 0.0f, det_half_spacing.z);
                det_bottom = det_affine.transform(offset_z1);
                det_top = det_affine.transform(offset_z2);
            }

            __device__ void compute_bounds_on_xz(const DeviceImage3D &image,
                                                 const Value3<T> &det_left, const Value3<T> &det_right,
                                                 const Value3<T> &det_bottom, const Value3<T> &det_top,
                                                 const Value3<T> &ray_direction, const T &pos_y,
                                                 IndexInt &bound_min, IndexInt &bound_max,
                                                 Value3<T> &pos_det_min, Value3<T> &pos_det_max)
            {
                T pos_x1 = interact_x_on_plane_y(det_left, ray_direction, pos_y);
                T pos_x2 = interact_x_on_plane_y(det_right, ray_direction, pos_y);
                T pos_z1 = interact_z_on_plane_y(det_bottom, ray_direction, pos_y);
                T pos_z2 = interact_z_on_plane_y(det_top, ray_direction, pos_y);
                _sort(pos_x1, pos_x2);
                _sort(pos_z1, pos_z2);

                // get the index of the voxel in the image which may intersect with the LOR
                bound_min = image.locate_index_by_pos(pos_x1, pos_y, pos_z1);
                bound_max = image.locate_index_by_pos(pos_x2, pos_y, pos_z2);
                pos_det_min = Value3<T>(pos_x1, pos_y, pos_z1);
                pos_det_max = Value3<T>(pos_x2, pos_y, pos_z2);
            }

            __device__ void compute_bounds_on_yz(const DeviceImage3D &image,
                                                 const Value3<T> &det_left, const Value3<T> &det_right,
                                                 const Value3<T> &det_bottom, const Value3<T> &det_top,
                                                 const Value3<T> &ray_direction, const T &pos_x,
                                                 IndexInt &bound_min, IndexInt &bound_max,
                                                 Value3<T> &pos_det_min, Value3<T> &pos_det_max)
            {
                // calculate the interaction point of lors on Y-Z plane
                T pos_y1 = interact_y_on_plane_x(det_left, ray_direction, pos_x);
                T pos_y2 = interact_y_on_plane_x(det_right, ray_direction, pos_x);
                T pos_z1 = interact_z_on_plane_x(det_bottom, ray_direction, pos_x);
                T pos_z2 = interact_z_on_plane_x(det_top, ray_direction, pos_x);
                _sort(pos_y1, pos_y2);
                _sort(pos_z1, pos_z2);

                // get the bound  point of the voxel in the image which may intersect with the LOR
                bound_min = image.locate_index_by_pos(pos_x, pos_y1, pos_z1);
                bound_max = image.locate_index_by_pos(pos_x, pos_y2, pos_z2);
                pos_det_min = Value3<T>(pos_x, pos_y1, pos_z1);
                pos_det_max = Value3<T>(pos_x, pos_y2, pos_z2);
            }

            __device__ T project_kernel(const DeviceImage3D &image,
                                        const Matrix4x4 &det_affine,
                                        const Coordinate3D &det_coordinate)
            {
                // initialize the projection value
                T value = 0;

                // get the ray direction
                auto ray_direction = det_affine.get_normal();

                // get the image size and spacing
                auto image_coordinate = image.coordinate;
                auto image_size = image_coordinate.size;
                auto image_spacing = image_coordinate.spacing;

                // get the detector size and spacing
                auto det_size = det_coordinate.size;
                auto det_half_spacing = det_coordinate.spacing / 2;

                // make an default origin for image and detector.
                const Value3<T> origin(0.0f, 0.0f, 0.0f);

                Value3<T> det_left, det_right, det_bottom, det_top;
                set_detector_bounds(det_affine, det_coordinate, det_left, det_right, det_bottom, det_top);

                // main axis is x
                if (abs(ray_direction.x) > abs(ray_direction.y))
                {
                    // calculate the interaction of lor and image pixel
                    const T interaction_length = compute_interaction_ratio_on_x(ray_direction);

                    // set the direction of ray to proceedding.
                    bool sign = ray_direction.x > 0 ? true : false;

                    // loop over x axis
                    for (size_t ix = 0; ix < image_size.x; ix++)
                    {
                        // if sign is negative, the ray is proceeding from right to left, so the index of image should be reoriented.
                        size_t reoriented_ix = sign ? ix : (image_size.x - ix - 1);
                        // calculat the current x center
                        const T pos_x = image.get_pos_x_by_index(reoriented_ix);

                        // ray do not go through the plane
                        if (ray_direction.x * (pos_x - det_left.x) < 0)
                            continue;
                        // TODO: time-of-flight weighting

                        // calculate the interaction point of lors on Y-Z plane
                        IndexInt bound_min, bound_max;
                        Value3<T> pos_det_min, pos_det_max;
                        compute_bounds_on_yz(image, det_left, det_right, det_bottom, det_top, ray_direction, pos_x, bound_min, bound_max, pos_det_min, pos_det_max);

                        // loop over yz plane to accumulate all the voxels
                        value += loop_yz_plane(image, bound_min, bound_max, pos_det_min, pos_det_max);
                    }
                    value *= interaction_length;
                }
                else // main axis is y
                {
                    const T interaction_length = compute_interaction_ratio_on_y(ray_direction);

                    // set the direction of ray to proceedding.
                    bool sign = ray_direction.y > 0 ? true : false;

                    // loop over y axis
                    for (size_t iy = 0; iy < image_size.y; iy++)
                    {
                        // if sign is negative, the ray is proceeding from right to left, so the index of image should be reoriented.
                        size_t reoriented_iy = sign ? iy : (image_size.y - iy - 1);

                        const T pos_y = image.get_pos_y_by_index(reoriented_iy);
                        // ray do not go through the plane
                        if (ray_direction.y * (pos_y - det_left.y) < 0)
                            continue;
                        IndexInt bound_min, bound_max;
                        Value3<T> pos_det_min, pos_det_max;
                        compute_bounds_on_xz(image, det_left, det_right, det_bottom, det_top, ray_direction, pos_y, bound_min, bound_max, pos_det_min, pos_det_max);
                        value += loop_xz_plane(image, bound_min, bound_max, pos_det_min, pos_det_max);
                    }

                    value *= interaction_length;
                }
                return value;
            }

            __global__ void project_global(const DeviceImage3D image,
                                           Value3<T> *pos_table_ptr, size_t num_rays_per_view,
                                           Matrix4x4 *aff_list_ptr, size_t num_views,
                                           const Coordinate3D det_coordinate, T *projection_value)
            {
                int ind_ray = blockIdx.x * blockDim.x + threadIdx.x;
                int ind_view = blockIdx.y * blockDim.y + threadIdx.y;
                if (ind_ray < num_rays_per_view && ind_view < num_views)
                {
                    // get the thread index
                    size_t thread_idx = ind_ray + ind_view * num_rays_per_view;

                    // get the affine matrix
                    Matrix4x4 affine_view = aff_list_ptr[ind_view]; // TODO: use reference or copy?
                    auto det_translation = Matrix4x4::make_translation(pos_table_ptr[ind_ray]);
                    auto composed_affine = affine_view.compose(det_translation);

                    // update the projection value with distance-driven ray tracing
                    projection_value[thread_idx] = project_kernel(image, composed_affine, det_coordinate);
                }
            }

            void project(const SPECT &scanner, Image3D &image, ProjectionValue &projection_value)
            {
                typedef Matrix4x4 Affine;

                // get the position lookup table on the detector
                auto collimator = scanner.get_collimator();
                auto det_coordinate = collimator.get_coordinate_prototype();
                auto lookup_table = det_coordinate.make_lookup_table();
                Value3<T> *lookup_table_ptr = thrust::raw_pointer_cast(lookup_table.data());
                auto image_device = image.to_device();

                // get the trajectory list for different detectors
                auto trajectory_list = scanner.get_trajectory_list();

                // get the output projection values.
                auto projection_value_ptr = thrust::raw_pointer_cast(projection_value.data());
                size_t ray_index = 0;

                // loop over trajectories(different detectors)
                size_t num_trajectories = trajectory_list.size();
                for (int i_traj = 0; i_traj < num_trajectories; i_traj++)
                {
                    auto current_traj = trajectory_list[i_traj];

                    // get the affine list for current trajectory
                    auto affine_list = current_traj.get_affine_list();

                    Affine *affine_list_ptr = thrust::raw_pointer_cast(affine_list.data());

                    // set the dimension of the grid and block
                    dim3 dimBlock(kThreadPerBlock_x, kThreadPerBlock_y);
                    const size_t num_rays_per_view = collimator.get_num_rays();
                    const size_t num_views = current_traj.num_views;
                    const size_t total_num_rays = num_rays_per_view * num_views;

                    // deal with the rays of a single trajectory
                    dim3 dimGrid((num_rays_per_view + kThreadPerBlock_x - 1) / kThreadPerBlock_x,
                                 (num_views + kThreadPerBlock_y - 1) / kThreadPerBlock_y);

                    // TODO: use texture memory to store the affine_list and pos_table to accelerate.
                    project_global<<<dimGrid, dimBlock>>>(image_device,
                                                          lookup_table_ptr, num_rays_per_view,
                                                          affine_list_ptr, num_views,
                                                          det_coordinate,
                                                          projection_value_ptr + ray_index);
                    ray_index += total_num_rays;
                }
                // lookup_table is local device storage.  The projection
                // kernels must finish before it is released on function exit.
                cudaDeviceSynchronize();
            }

            __device__ T atten_project_kernel(const DeviceImage3D &image,
                                              const DeviceImage3D &atten_map,
                                              const Matrix4x4 &det_affine,
                                              const Coordinate3D &det_coordinate)
            {
                // initialize the projection value
                T value = 0;

                // get the ray direction
                auto ray_direction = det_affine.get_normal();

                // get the image size and spacing
                auto image_coordinate = image.coordinate;
                auto image_size = image_coordinate.size;
                auto image_spacing = image_coordinate.spacing;

                // get the detector size and spacing
                // auto det_size = det_coordinate.size;
                auto det_half_spacing = det_coordinate.spacing / 2;

                // make an default origin for image and detector.
                const Value3<T> origin(0.0f, 0.0f, 0.0f);

                // compute detector bounds of current ray
                Value3<T> det_left, det_right, det_bottom, det_top;
                set_detector_bounds(det_affine, det_coordinate, det_left, det_right, det_bottom, det_top);

                T atten_integral = 0;

                // main axis is x
                if (abs(ray_direction.x) > abs(ray_direction.y))
                {
                    // calculate the interaction of lor and image pixel
                    const T interaction_length = compute_interaction_ratio_on_x(ray_direction) * image_spacing.x;

                    // set the direction of ray to proceedding.
                    bool sign = ray_direction.x > 0 ? true : false;

                    // loop over x axis
                    for (size_t ix = 0; ix < image_size.x; ix++)
                    {
                        // if sign is negative, the ray is proceeding from right to left, so the index of image should be reoriented.
                        size_t reoriented_ix = sign ? ix : (image_size.x - ix - 1);
                        // calculate the current x center
                        const T pos_x = image.get_pos_x_by_index(reoriented_ix);

                        // ray do not go through the plane
                        if (ray_direction.x * (pos_x - det_left.x) < 0)
                            continue;
                        // TODO: time-of-flight weighting

                        // calculate the interaction point of lors on Y-Z plane
                        IndexInt bound_min, bound_max;
                        Value3<T> pos_det_min, pos_det_max;
                        compute_bounds_on_yz(image, det_left, det_right, det_bottom, det_top, ray_direction, pos_x, bound_min, bound_max, pos_det_min, pos_det_max);
                        // loop over yz plane to accumulate all the voxels
                        value += loop_yz_plane(image, atten_map, atten_integral, interaction_length, bound_min, bound_max, pos_det_min, pos_det_max);
                    }
                }
                else // main axis is y
                {
                    const T interaction_length = compute_interaction_ratio_on_y(ray_direction) * image_spacing.y;

                    // set the direction of ray to proceedding.
                    bool sign = ray_direction.y > 0 ? true : false;

                    // loop over y axis
                    for (size_t iy = 0; iy < image_size.y; iy++)
                    {
                        // if sign is negative, the ray is proceeding from right to left, so the index of image should be reoriented.
                        size_t reoriented_iy = sign ? iy : (image_size.y - iy - 1);

                        const T pos_y = image.get_pos_y_by_index(reoriented_iy);
                        // ray do not go through the plane
                        if (ray_direction.y * (pos_y - det_left.y) < 0)
                            continue;
                        IndexInt bound_min, bound_max;
                        Value3<T> pos_det_min, pos_det_max;
                        compute_bounds_on_xz(image, det_left, det_right, det_bottom, det_top, ray_direction, pos_y, bound_min, bound_max, pos_det_min, pos_det_max);
                        value += loop_xz_plane(image, atten_map, atten_integral, interaction_length, bound_min, bound_max, pos_det_min, pos_det_max);
                    }
                }
                return value;
            }

            __global__ void atten_project_global(const DeviceImage3D image,
                                                 const DeviceImage3D atten_map,
                                                 Value3<T> *pos_table_ptr, size_t num_rays_per_view,
                                                 Matrix4x4 *aff_list_ptr, size_t num_views,
                                                 const Coordinate3D det_coordinate, T *projection_value)
            {
                int ind_ray = blockIdx.x * blockDim.x + threadIdx.x;
                int ind_view = blockIdx.y * blockDim.y + threadIdx.y;
                if (ind_ray < num_rays_per_view && ind_view < num_views)
                {
                    // get the thread index
                    int thread_idx = ind_ray + ind_view * num_rays_per_view;

                    // get the affine matrix of the projected rays
                    Matrix4x4 affine_view = aff_list_ptr[ind_view]; // TODO: use reference or copy?
                    auto det_translation = Matrix4x4::make_translation(pos_table_ptr[ind_ray]);
                    auto composed_affine = affine_view.compose(det_translation);

                    // update the projection value with distance-driven ray tracing
                    projection_value[thread_idx] = atten_project_kernel(image, atten_map, composed_affine, det_coordinate);
                }
            }

            void atten_project(const SPECT &scanner,
                               Image3D &image,
                               Image3D &atten_map,
                               ProjectionValue &projection_value)
            {
                typedef Matrix4x4 Affine;

                // get the position lookup table on the detector
                auto collimator = scanner.get_collimator();
                auto det_coordinate = collimator.get_coordinate_prototype();
                auto lookup_table = det_coordinate.make_lookup_table();
                Value3<T> *lookup_table_ptr = thrust::raw_pointer_cast(lookup_table.data());

                // get the image and atten map on the device
                auto image_device = image.to_device();
                auto atten_map_device = atten_map.to_device();

                // get the trajectory list for different detectors
                auto trajectory_list = scanner.get_trajectory_list();

                // set the output projection values.
                T *projection_value_ptr = thrust::raw_pointer_cast(projection_value.data());

                // set the index of the projection value for different trajectories.
                size_t ray_index = 0;

                // loop over trajectories(different detectors)
                size_t num_trajectories = trajectory_list.size();
                for (int i_traj = 0; i_traj < num_trajectories; i_traj++)
                {
                    auto current_traj = trajectory_list[i_traj];
                    // get the affine list for current trajectory
                    auto affine_list = current_traj.get_affine_list();
                    Affine *affine_list_ptr = thrust::raw_pointer_cast(affine_list.data());
                    const size_t num_rays_per_view = collimator.get_num_rays();
                    const size_t num_views = current_traj.num_views;
                    const size_t total_num_rays = num_rays_per_view * num_views;

                    // set the dimension of the parallel grid and block
                    dim3 dimBlock(kThreadPerBlock_x, kThreadPerBlock_y);
                    dim3 dimGrid((num_rays_per_view + kThreadPerBlock_x - 1) / kThreadPerBlock_x,
                                 (num_views + kThreadPerBlock_y - 1) / kThreadPerBlock_y);

                    // process the rays of a single trajectory
                    atten_project_global<<<dimGrid, dimBlock>>>(image_device, atten_map_device,
                                                                lookup_table_ptr, num_rays_per_view,
                                                                affine_list_ptr, num_views,
                                                                det_coordinate,
                                                                projection_value_ptr + ray_index);
                    ray_index += total_num_rays;
                }
                // See generic_project: retain the local lookup table until
                // all asynchronous projection kernels have completed.
                cudaDeviceSynchronize();
            }

            __device__ void backproject_kernel(DeviceImage3D &image,
                                               const Matrix4x4 &det_affine,
                                               const Coordinate3D &det_coordinate,
                                               T value)
            {
                auto ray_direction = det_affine.get_normal();
                auto image_coordinate = image.get_coordinate();
                auto image_size = image_coordinate.size;
                auto image_spacing = image_coordinate.spacing;

                auto det_size = det_coordinate.size;
                auto det_half_spacing = det_coordinate.spacing / 2;

                const Value3<T> origin(0, 0, 0);

                Value3<T> det_left, det_right, det_bottom, det_top;
                set_detector_bounds(det_affine, det_coordinate, det_left, det_right, det_bottom, det_top);

                if (abs(ray_direction.x) > abs(ray_direction.y))
                {
                    // calculate the interaction of lor and image pixel
                    const T interaction_length = compute_interaction_ratio_on_x(ray_direction);

                    // set the direction of ray to proceedding.
                    bool sign = ray_direction.x > 0 ? true : false;

                    // loop over x axis
                    for (size_t ix = 0; ix < image_size.x; ix++)
                    {
                        // if sign is negative, the ray is proceeding from right to left, so the index of image should be reoriented.
                        size_t reoriented_ix = sign ? ix : (image_size.x - ix - 1);
                        // calculat the current x center
                        const T pos_x = image.get_pos_x_by_index(reoriented_ix);

                        // ray do not go through the plane
                        if (ray_direction.x * (pos_x - det_left.x) < 0)
                            continue;

                        IndexInt bound_min, bound_max;
                        Value3<T> pos_det_min, pos_det_max;
                        compute_bounds_on_yz(image, det_left, det_right, det_bottom, det_top, ray_direction, pos_x, bound_min, bound_max, pos_det_min, pos_det_max);

                        // loop over yz plane to accumulate all the voxels
                        loop_yz_plane_bp(value, image, bound_min, bound_max, pos_det_min, pos_det_max);
                    }
                }
                else // main axis is y
                {
                    const T interaction_length = compute_interaction_ratio_on_y(ray_direction);

                    // set the direction of ray to proceedding.
                    bool sign = ray_direction.y > 0 ? true : false;

                    // loop over y axis
                    for (size_t iy = 0; iy < image_size.y; iy++)
                    {
                        // if sign is negative, the ray is proceeding from right to left, so the index of image should be reoriented.
                        size_t reoriented_iy = sign ? iy : (image_size.y - iy - 1);

                        const T pos_y = image.get_pos_y_by_index(reoriented_iy);
                        // ray do not go through the plane
                        if (ray_direction.y * (pos_y - det_left.y) < 0)
                            continue;

                        IndexInt bound_min, bound_max;
                        Value3<T> pos_det_min, pos_det_max;
                        compute_bounds_on_xz(image, det_left, det_right, det_bottom, det_top, ray_direction, pos_y, bound_min, bound_max, pos_det_min, pos_det_max);
                        loop_xz_plane_bp(value, image, bound_min, bound_max, pos_det_min, pos_det_max);
                    }
                }
            }

            __global__ void backproject_global(DeviceImage3D image,
                                               Value3<T> *pos_table_ptr, size_t num_rays_per_view,
                                               Matrix4x4 *aff_list_ptr, size_t num_views,
                                               const Coordinate3D det_coordinate, T *projection_value)
            {
                int ind_ray = blockIdx.x * blockDim.x + threadIdx.x;
                int ind_view = blockIdx.y * blockDim.y + threadIdx.y;
                if (ind_ray < num_rays_per_view && ind_view < num_views)
                {
                    // get the thread index
                    int thread_idx = ind_ray + ind_view * num_rays_per_view;

                    // get the affine matrix
                    Matrix4x4 affine_view = aff_list_ptr[ind_view]; // TODO: use reference or copy?
                    auto det_translation = Matrix4x4::make_translation(pos_table_ptr[ind_ray]);
                    auto composed_affine = affine_view.compose(det_translation);

                    backproject_kernel(image, composed_affine, det_coordinate, projection_value[thread_idx]);
                }
            }

            void backproject(const SPECT &scanner, Image3D &image, ProjectionValue &projection_value)
            {
                typedef Matrix4x4 Affine;
                auto collimator = scanner.get_collimator();
                auto det_coordinate = collimator.get_coordinate_prototype();
                auto lookup_table = det_coordinate.make_lookup_table();
                Value3<T> *lookup_table_ptr = thrust::raw_pointer_cast(lookup_table.data());
                auto image_device = image.to_device();

                auto trajectory_list = scanner.get_trajectory_list();
                T *projection_value_ptr = thrust::raw_pointer_cast(projection_value.data());
                size_t ray_index = 0;

                size_t num_trajectories = trajectory_list.size();
                for (int i_traj = 0; i_traj < num_trajectories; i_traj++)
                {
                    auto current_traj = trajectory_list[i_traj];

                    // get the affine list for current trajectory
                    auto affine_list = current_traj.get_affine_list();
                    Affine *affine_list_ptr = thrust::raw_pointer_cast(affine_list.data());
                    dim3 dimBlock(kThreadPerBlock_x, kThreadPerBlock_y);

                    const size_t num_rays_per_view = collimator.get_num_rays();
                    const size_t num_views = current_traj.num_views;
                    const size_t total_num_rays = num_rays_per_view * num_views;
                    // deal with the rays of a single trajectory
                    dim3 dimGrid((num_rays_per_view + kThreadPerBlock_x - 1) / kThreadPerBlock_x,
                                 (num_views + kThreadPerBlock_y - 1) / kThreadPerBlock_y);
                    backproject_global<<<dimGrid, dimBlock>>>(image_device, lookup_table_ptr, num_rays_per_view, affine_list_ptr, num_views, det_coordinate, projection_value_ptr + ray_index);
                    ray_index += total_num_rays;
                }
            }

            __device__ void atten_backproject_kernel(DeviceImage3D &image, DeviceImage3D &atten_map,
                                                     const Matrix4x4 &det_affine,
                                                     const Coordinate3D &det_coordinate,
                                                     T value)
            {
                auto ray_direction = det_affine.get_normal();
                auto image_coordinate = image.coordinate;
                auto image_size = image_coordinate.size;
                auto image_spacing = image_coordinate.spacing;

                auto det_size = det_coordinate.size;
                auto det_half_spacing = det_coordinate.spacing / 2;

                // const Value3<T> origin(0, 0, 0);

                Value3<T> det_left, det_right, det_bottom, det_top;
                set_detector_bounds(det_affine, det_coordinate, det_left, det_right, det_bottom, det_top);

                T atten_integral = 0.0;
                if (abs(ray_direction.x) > abs(ray_direction.y))
                {
                    // calculate the interaction of lor and image pixel
                    const T interaction_length = compute_interaction_ratio_on_x(ray_direction) * image_spacing.x;

                    // set the direction of ray to proceedding.
                    bool sign = ray_direction.x > 0 ? true : false;

                    // loop over x axis
                    for (size_t ix = 0; ix < image_size.x; ix++)
                    {
                        // if sign is negative, the ray is proceeding from right to left, so the index of image should be reoriented.
                        size_t reoriented_ix = sign ? ix : (image_size.x - ix - 1);
                        // calculat the current x center
                        const T pos_x = image.get_pos_x_by_index(reoriented_ix);

                        // ray do not go through the plane
                        if (ray_direction.x * (pos_x - det_left.x) < 0)
                            continue;
                        // TODO: time-of-flight weighting

                        // calculate the interaction point of lors on Y-Z plane
                        IndexInt bound_min, bound_max;
                        Value3<T> pos_det_min, pos_det_max;
                        compute_bounds_on_yz(image, det_left, det_right, det_bottom, det_top, ray_direction, pos_x, bound_min, bound_max, pos_det_min, pos_det_max);

                        // loop over yz plane to accumulate all the voxels
                        loop_yz_plane_bp(value, image, atten_map, atten_integral, interaction_length, bound_min, bound_max, pos_det_min, pos_det_max);
                    }
                }
                else // main axis is y
                {
                    const T interaction_length = compute_interaction_ratio_on_y(ray_direction) * image_spacing.y;

                    // set the direction of ray to proceedding.
                    bool sign = ray_direction.y > 0 ? true : false;

                    // loop over y axis
                    for (size_t iy = 0; iy < image_size.y; iy++)
                    {
                        // if sign is negative, the ray is proceeding from right to left, so the index of image should be reoriented.
                        size_t reoriented_iy = sign ? iy : (image_size.y - iy - 1);

                        const T pos_y = image.get_pos_y_by_index(reoriented_iy);
                        // ray do not go through the plane
                        if (ray_direction.y * (pos_y - det_left.y) < 0)
                            continue;

                        IndexInt bound_min, bound_max;
                        Value3<T> pos_det_min, pos_det_max;
                        compute_bounds_on_xz(image, det_left, det_right, det_bottom, det_top, ray_direction, pos_y, bound_min, bound_max, pos_det_min, pos_det_max);
                        loop_xz_plane_bp(value, image, atten_map, atten_integral, interaction_length, bound_min, bound_max, pos_det_min, pos_det_max);
                    }
                }
            }

            __global__ void atten_backproject_global(DeviceImage3D image, DeviceImage3D atten_map,
                                                     Value3<T> *pos_table_ptr, size_t num_rays_per_view,
                                                     Matrix4x4 *aff_list_ptr, size_t num_views,
                                                     const Coordinate3D det_coordinate, T *projection_value)
            {
                int ind_ray = blockIdx.x * blockDim.x + threadIdx.x;
                int ind_view = blockIdx.y * blockDim.y + threadIdx.y;
                if (ind_ray < num_rays_per_view && ind_view < num_views)
                {
                    // get the thread index
                    int thread_idx = ind_ray + ind_view * num_rays_per_view;

                    // get the affine matrix
                    Matrix4x4 affine_view = aff_list_ptr[ind_view]; // TODO: use reference or copy?
                    auto det_translation = Matrix4x4::make_translation(pos_table_ptr[ind_ray]);
                    auto composed_affine = affine_view.compose(det_translation);

                    atten_backproject_kernel(image, atten_map, composed_affine, det_coordinate, projection_value[thread_idx]);
                }
            }

            void atten_backproject(const SPECT &scanner,
                                   Image3D &image,
                                   Image3D &atten_map,
                                   ProjectionValue &projection_value)
            {
                typedef Matrix4x4 Affine;
                auto collimator = scanner.get_collimator();
                auto det_coordinate = collimator.get_coordinate_prototype();
                auto lookup_table = det_coordinate.make_lookup_table();
                Value3<T> *lookup_table_ptr = thrust::raw_pointer_cast(lookup_table.data());
                auto image_device = image.to_device();
                auto atten_map_device = atten_map.to_device();

                auto trajectory_list = scanner.get_trajectory_list();
                T *projection_value_ptr = thrust::raw_pointer_cast(projection_value.data());
                size_t ray_index = 0;

                size_t num_trajectories = trajectory_list.size();
                for (int i_traj = 0; i_traj < num_trajectories; i_traj++)
                {
                    auto current_traj = trajectory_list[i_traj];

                    // get the affine list for current trajectory
                    auto affine_list = current_traj.get_affine_list();
                    Affine *affine_list_ptr = thrust::raw_pointer_cast(affine_list.data());
                    dim3 dimBlock(kThreadPerBlock_x, kThreadPerBlock_y);
                    const size_t num_rays_per_view = collimator.get_num_rays();
                    const size_t num_views = current_traj.num_views;
                    const size_t total_num_rays = num_rays_per_view * num_views;
                    dim3 dimGrid((num_rays_per_view + kThreadPerBlock_x - 1) / kThreadPerBlock_x,
                                 (num_views + kThreadPerBlock_y - 1) / kThreadPerBlock_y);
                    atten_backproject_global<<<dimGrid, dimBlock>>>(image_device, atten_map_device,
                                                                    lookup_table_ptr, num_rays_per_view,
                                                                    affine_list_ptr, num_views,
                                                                    det_coordinate, projection_value_ptr + ray_index);
                    ray_index += total_num_rays;
                }
            }

            __global__ void cdr_project_global(const DeviceImage3D image,
                                               const DeviceImage3D atten_map,
                                               Value3<T> *pos_table_ptr, size_t num_rays_per_view,
                                               Matrix4x4 *aff_list_ptr, size_t num_views,
                                               const Coordinate3D det_coordinate,
                                               const GaussianCDR cdr_model,
                                               T *projection_value)
            {
                int ind_ray = blockIdx.x * blockDim.x + threadIdx.x;
                int ind_view = blockIdx.y * blockDim.y + threadIdx.y;
                if (ind_ray < num_rays_per_view && ind_view < num_views)
                {
                    // int thread_idx = ind_ray + ind_view * num_rays_per_view;

                    // get the affine matrix
                    Matrix4x4 affine_view = aff_list_ptr[ind_view];
                    auto det_translation = Matrix4x4::make_translation(pos_table_ptr[ind_ray]);
                    auto composed_affine = affine_view.compose(det_translation);

                    // project the image to the detector plane with cdr weighting
                    T value = cdr_project_kernel(image, atten_map, composed_affine, det_coordinate,
                                                 cdr_model, ind_ray, projection_value + ind_view * num_rays_per_view);

                    // __syncthreads();
                    //
                }
            }

            __device__ T cdr_project_kernel(const DeviceImage3D &image,
                                            const DeviceImage3D &atten_map,
                                            const Matrix4x4 &det_affine,
                                            const Coordinate3D &det_coordinate,
                                            const GaussianCDR &cdr_model,
                                            const int &ind_ray,
                                            T *projection_value_current_view)
            {
                // create the accumulated value for CDR weighted value and set to zero.
                T accumulated_cdr_values[kMaxMapSize * kMaxMapSize] = {0.0f};
                T value = 0;
                auto ray_direction = det_affine.get_normal();
                auto image_size = image.coordinate.size;
                auto image_spacing = image.coordinate.spacing;

                Value3<T> det_left, det_right, det_bottom, det_top;
                set_detector_bounds(det_affine, det_coordinate, det_left, det_right, det_bottom, det_top);
                T atten_integral = 0;
                T pixel_to_detector_distance = 0;
                if (abs(ray_direction.x) >= abs(ray_direction.y))
                {
                    // calculate the interaction of lor and image pixel
                    const T interaction_length = compute_interaction_ratio_on_x(ray_direction) * image_spacing.x;

                    // set the direction of ray to proceedding.
                    bool sign = ray_direction.x > 0 ? true : false;

                    // loop over x axis
                    for (size_t ix = 0; ix < image_size.x; ix++)
                    {
                        // if sign is negative, the ray is proceeding from right to left, so the index of image should be reoriented.
                        size_t reoriented_ix = sign ? ix : (image_size.x - ix - 1);
                        // calculat the current x center of the image pixel
                        const T pos_x = image.get_pos_x_by_index(reoriented_ix);

                        // the ray does not go through the plane (this is a loose restriction, can be optimized)
                        if (ray_direction.x * (pos_x - det_left.x) < 0)
                            continue;

                        // calculate the bound of the voxel in the image on y-z plane which may intersect with the ray
                        IndexInt bound_min, bound_max;
                        Value3<T> pos_det_min, pos_det_max;
                        compute_bounds_on_yz(image, det_left, det_right, det_bottom, det_top,
                                             ray_direction, pos_x, bound_min, bound_max, pos_det_min, pos_det_max);

                        // loop over yz plane to accumulate all the voxels
                        value = loop_yz_plane(image, atten_map, atten_integral, interaction_length,
                                              bound_min, bound_max, pos_det_min, pos_det_max);

                        // compute the distance from the plane to the detector
                        // TODO: current method is not accurate, all the pixels in the plane are assumed to be the same distance from the detector.
                        // if the detector spacing is small
                        pixel_to_detector_distance = abs((pos_x - (det_left.x + det_right.x) / 2) / ray_direction.x);

                        // accumulate the value with cdr model.
                        cdr_model.project(det_coordinate, pixel_to_detector_distance, ind_ray, value, accumulated_cdr_values);
                    }
                }
                else // main axis is y
                {
                    const T interaction_length = compute_interaction_ratio_on_y(ray_direction) * image_spacing.y;

                    // set the direction of ray to proceedding.
                    bool sign = ray_direction.y > 0 ? true : false;

                    // loop over y axis
                    for (size_t iy = 0; iy < image_size.y; iy++)
                    {
                        // if sign is negative, the ray is proceeding from right to left, so the index of image should be reoriented.
                        size_t reoriented_iy = sign ? iy : (image_size.y - iy - 1);

                        const T pos_y = image.get_pos_y_by_index(reoriented_iy);
                        // ray do not go through the plane
                        if (ray_direction.y * (pos_y - det_left.y) < 0)
                            continue;

                        // calculate the bound of the voxel in the image on x-z plane which may intersect with the ray
                        IndexInt bound_min, bound_max;
                        Value3<T> pos_det_min, pos_det_max;
                        compute_bounds_on_xz(image, det_left, det_right, det_bottom, det_top,
                                             ray_direction, pos_y, bound_min, bound_max, pos_det_min, pos_det_max);

                        value = loop_xz_plane(image, atten_map, atten_integral, interaction_length,
                                              bound_min, bound_max, pos_det_min, pos_det_max);

                        // compute the distance from the plane to the detector
                        pixel_to_detector_distance = abs((pos_y - (det_left.y + det_right.y) / 2) / ray_direction.y);

                        cdr_model.project(det_coordinate, pixel_to_detector_distance, ind_ray, value, accumulated_cdr_values);
                    }
                }
                cdr_model.update_projection(det_coordinate, ind_ray, accumulated_cdr_values, projection_value_current_view);
                return value;
            }

            void cdr_project(const SPECT &scanner,
                             const GaussianCDR &cdr_model,
                             Image3D &image,
                             Image3D &atten_map,
                             ProjectionValue &projection_value)
            {
                typedef Matrix4x4 Affine;

                // get the position lookup table on the detector
                auto collimator = scanner.get_collimator();
                auto det_coordinate = collimator.get_coordinate_prototype();
                auto lookup_table = det_coordinate.make_lookup_table();
                Value3<T> *lookup_table_ptr = thrust::raw_pointer_cast(lookup_table.data());

                // get the image and atten map on the device
                auto image_device = image.to_device();
                auto atten_map_device = atten_map.to_device();

                // get the trajectory list for different detectors
                auto trajectory_list = scanner.get_trajectory_list();

                // Note: clear the value of the projection because it is accumulated.
                thrust::fill(projection_value.begin(), projection_value.end(), T(0));
                // set the output projection values.
                T *projection_value_ptr = thrust::raw_pointer_cast(projection_value.data());

                // set the index of the projection value for different trajectories.
                size_t ray_index = 0;

                // loop over trajectories(different detectors)
                size_t num_trajectories = trajectory_list.size();
                for (int i_traj = 0; i_traj < num_trajectories; i_traj++)
                {
                    // std::cout << "Processing trajectory " << i_traj << std::endl;
                    // get the affine list for current trajectory
                    auto current_traj = trajectory_list[i_traj];
                    auto affine_list = current_traj.get_affine_list();
                    Affine *affine_list_ptr = thrust::raw_pointer_cast(affine_list.data());

                    // set the dimension of the grid and block
                    dim3 dimBlock(kThreadPerBlock_x, kThreadPerBlock_y);
                    const size_t num_rays_per_view = collimator.get_num_rays();
                    const size_t num_views = current_traj.num_views;
                    const size_t total_num_rays = num_rays_per_view * num_views;

                    // deal with the rays of a single trajectory
                    dim3 dimGrid((num_rays_per_view + kThreadPerBlock_x - 1) / kThreadPerBlock_x,
                                 (num_views + kThreadPerBlock_y - 1) / kThreadPerBlock_y);

                    // calculate the projection value with CDR model
                    cdr_project_global<<<dimGrid, dimBlock>>>(image_device, atten_map_device,
                                                              lookup_table_ptr, num_rays_per_view,
                                                              affine_list_ptr, num_views,
                                                              det_coordinate, cdr_model,
                                                              projection_value_ptr + ray_index);
                    // cudaDeviceSynchronize();
                    ray_index += total_num_rays;
                }
            }

            __device__ void cdr_backproject_kernel(DeviceImage3D &image,
                                                   DeviceImage3D &atten_map,
                                                   const Matrix4x4 &det_affine,
                                                   const Coordinate3D &det_coordinate,
                                                   const GaussianCDR &cdr_model,
                                                   const T *projection_value_current_view,
                                                   const int ind_ray)
            {
                auto ray_direction = det_affine.get_normal();
                auto image_coordinate = image.coordinate;
                auto image_size = image_coordinate.size;
                auto image_spacing = image_coordinate.spacing;

                auto det_size = det_coordinate.size;
                auto det_half_spacing = det_coordinate.spacing / 2;

                const Value3<T> origin(0, 0, 0);

                Value3<T> det_left, det_right, det_bottom, det_top;
                set_detector_bounds(det_affine, det_coordinate, det_left, det_right, det_bottom, det_top);

                T atten_integral = 0;
                T pixel_to_detector_distance = 0;
                if (abs(ray_direction.x) >= abs(ray_direction.y))
                {
                    // calculate the interaction of lor and image pixel
                    const T interaction_length = compute_interaction_ratio_on_x(ray_direction) * image_spacing.x;

                    // set the direction of ray to proceedding.
                    bool sign = ray_direction.x > 0 ? true : false;

                    // loop over x axis
                    for (size_t ix = 0; ix < image_size.x; ix++)
                    {
                        // if sign is negative, the ray is proceeding from right to left, so the index of image should be reoriented.
                        size_t reoriented_ix = sign ? ix : (image_size.x - ix - 1);
                        // calculat the current x center
                        const T pos_x = image.get_pos_x_by_index(reoriented_ix);

                        // ray do not go through the plane
                        if (ray_direction.x * (pos_x - det_left.x) < 0)
                            continue;

                        IndexInt bound_min, bound_max;
                        Value3<T> pos_det_min, pos_det_max;
                        compute_bounds_on_yz(image, det_left, det_right, det_bottom, det_top, ray_direction, pos_x, bound_min, bound_max, pos_det_min, pos_det_max);
                        pixel_to_detector_distance = abs((pos_x - (det_left.x + det_right.x) / 2) / ray_direction.x);

                        T value = cdr_model.convolve_projection(det_coordinate, pixel_to_detector_distance, ind_ray, projection_value_current_view);

                        loop_yz_plane_bp(value, image, atten_map, atten_integral, interaction_length, bound_min, bound_max, pos_det_min, pos_det_max);
                    }
                }
                else
                {
                    const T interaction_length = compute_interaction_ratio_on_y(ray_direction) * image_spacing.y;

                    // set the direction of ray to proceedding.
                    bool sign = ray_direction.y > 0 ? true : false;

                    // loop over y axis
                    for (size_t iy = 0; iy < image_size.y; iy++)
                    {
                        // if sign is negative, the ray is proceeding from right to left, so the index of image should be reoriented.
                        size_t reoriented_iy = sign ? iy : (image_size.y - iy - 1);

                        const T pos_y = image.get_pos_y_by_index(reoriented_iy);
                        // ray do not go through the plane
                        if (ray_direction.y * (pos_y - det_left.y) < 0)
                            continue;

                        IndexInt bound_min, bound_max;
                        Value3<T> pos_det_min, pos_det_max;
                        compute_bounds_on_xz(image, det_left, det_right, det_bottom, det_top, ray_direction, pos_y, bound_min, bound_max, pos_det_min, pos_det_max);
                        pixel_to_detector_distance = abs((pos_y - (det_left.y + det_right.y) / 2) / ray_direction.y);

                        T value = cdr_model.convolve_projection(det_coordinate, pixel_to_detector_distance, ind_ray, projection_value_current_view);

                        loop_xz_plane_bp(value, image, atten_map, atten_integral, interaction_length, bound_min, bound_max, pos_det_min, pos_det_max);
                    }
                }
            }

            __global__ void cdr_backproject_global(DeviceImage3D image, DeviceImage3D atten_map,
                                                   Value3<T> *pos_table_ptr, size_t num_rays_per_view,
                                                   Matrix4x4 *aff_list_ptr, size_t num_views,
                                                   const Coordinate3D det_coordinate,
                                                   const GaussianCDR cdr_model, T *projection_value)
            {
                int ind_ray = blockIdx.x * blockDim.x + threadIdx.x;
                int ind_view = blockIdx.y * blockDim.y + threadIdx.y;
                if (ind_ray < num_rays_per_view && ind_view < num_views)
                {
                    // get the thread index
                    // int thread_idx = ind_ray + ind_view * num_rays_per_view;

                    // get the affine matrix
                    Matrix4x4 affine_view = aff_list_ptr[ind_view]; // TODO: use reference or copy?
                    auto det_translation = Matrix4x4::make_translation(pos_table_ptr[ind_ray]);
                    auto composed_affine = affine_view.compose(det_translation);

                    cdr_backproject_kernel(image, atten_map, composed_affine, det_coordinate,
                                           cdr_model, projection_value + ind_view * num_rays_per_view, ind_ray);
                }
            }

            void cdr_backproject(const SPECT &scanner,
                                 const GaussianCDR &cdr_model,
                                 Image3D &image,
                                 Image3D &atten_map,
                                 ProjectionValue &projection_value)
            {
                typedef Matrix4x4 Affine;
                auto collimator = scanner.get_collimator();
                auto det_coordinate = collimator.get_coordinate_prototype();
                auto lookup_table = det_coordinate.make_lookup_table();
                Value3<T> *lookup_table_ptr = thrust::raw_pointer_cast(lookup_table.data());
                auto image_device = image.to_device();
                auto atten_map_device = atten_map.to_device();

                auto trajectory_list = scanner.get_trajectory_list();
                T *projection_value_ptr = thrust::raw_pointer_cast(projection_value.data());
                size_t ray_index = 0;

                size_t num_trajectories = trajectory_list.size();
                for (int i_traj = 0; i_traj < num_trajectories; i_traj++)
                {
                    auto current_traj = trajectory_list[i_traj];

                    // get the affine list for current trajectory
                    auto affine_list = current_traj.get_affine_list();
                    Affine *affine_list_ptr = thrust::raw_pointer_cast(affine_list.data());
                    dim3 dimBlock(kThreadPerBlock_x, kThreadPerBlock_y);
                    const size_t num_rays_per_view = collimator.get_num_rays();
                    const size_t num_views = current_traj.num_views;
                    const size_t total_num_rays = num_rays_per_view * num_views;
                    dim3 dimGrid((num_rays_per_view + kThreadPerBlock_x - 1) / kThreadPerBlock_x,
                                 (num_views + kThreadPerBlock_y - 1) / kThreadPerBlock_y);
                    cdr_backproject_global<<<dimGrid, dimBlock>>>(image_device, atten_map_device,
                                                                  lookup_table_ptr, num_rays_per_view,
                                                                  affine_list_ptr, num_views,
                                                                  det_coordinate, cdr_model,
                                                                  projection_value_ptr + ray_index);
                    ray_index += total_num_rays;
                }
            }

            __device__ T generic_project_kernel(const DeviceImage3D &image,
                                                const DeviceImage3D &atten_map,
                                                const Matrix4x4 &det_affine,
                                                const Coordinate3D &det_coordinate)
            {
                // initialize the projection value
                T value = 0;

                // get the ray direction
                auto ray_direction = det_affine.get_normal();

                // get the image size and spacing
                auto image_coordinate = image.coordinate;
                auto image_size = image_coordinate.size;
                auto image_spacing = image_coordinate.spacing;

                // get the detector size and spacing
                // auto det_size = det_coordinate.size;
                auto det_half_spacing = det_coordinate.spacing / 2;

                // make an default origin for image and detector.
                const Value3<T> origin(0.0f, 0.0f, 0.0f);

                // compute detector bounds of current ray
                Value3<T> det_left, det_right, det_bottom, det_top;
                set_detector_bounds(det_affine, det_coordinate, det_left, det_right, det_bottom, det_top);

                T atten_integral = 0;

                // main axis is x
                if (abs(ray_direction.x) > abs(ray_direction.y))
                {
                    // calculate the interaction of lor and image pixel
                    const T interaction_length = compute_interaction_ratio_on_x(ray_direction) * image_spacing.x;

                    // set the direction of ray to proceedding.
                    bool sign = ray_direction.x > 0 ? true : false;

                    // loop over x axis
                    for (size_t ix = 0; ix < image_size.x; ix++)
                    {
                        // if sign is negative, the ray is proceeding from right to left, so the index of image should be reoriented.
                        size_t reoriented_ix = sign ? ix : (image_size.x - ix - 1);
                        // calculate the current x center
                        const T pos_x = image.get_pos_x_by_index(reoriented_ix);

                        // ray do not go through the plane
                        if (ray_direction.x * (pos_x - det_left.x) < 0)
                            continue;
                        // TODO: time-of-flight weighting

                        // calculate the interaction point of lors on Y-Z plane
                        IndexInt bound_min, bound_max;
                        Value3<T> pos_det_min, pos_det_max;
                        compute_bounds_on_yz(image, det_left, det_right, det_bottom, det_top, ray_direction, pos_x, bound_min, bound_max, pos_det_min, pos_det_max);
                        // loop over yz plane to accumulate all the voxels
                        value += loop_yz_plane(image, atten_map, atten_integral, interaction_length, bound_min, bound_max, pos_det_min, pos_det_max);
                    }
                }
                else // main axis is y
                {
                    const T interaction_length = compute_interaction_ratio_on_y(ray_direction) * image_spacing.y;

                    // set the direction of ray to proceedding.
                    bool sign = ray_direction.y > 0 ? true : false;

                    // loop over y axis
                    for (size_t iy = 0; iy < image_size.y; iy++)
                    {
                        // if sign is negative, the ray is proceeding from right to left, so the index of image should be reoriented.
                        size_t reoriented_iy = sign ? iy : (image_size.y - iy - 1);

                        const T pos_y = image.get_pos_y_by_index(reoriented_iy);
                        // ray do not go through the plane
                        if (ray_direction.y * (pos_y - det_left.y) < 0)
                            continue;
                        IndexInt bound_min, bound_max;
                        Value3<T> pos_det_min, pos_det_max;
                        compute_bounds_on_xz(image, det_left, det_right, det_bottom, det_top, ray_direction, pos_y, bound_min, bound_max, pos_det_min, pos_det_max);
                        value += loop_xz_plane(image, atten_map, atten_integral, interaction_length, bound_min, bound_max, pos_det_min, pos_det_max);
                    }
                }
                return value;
            }

            __global__ void generic_project_global(const DeviceImage3D image,
                                                   const DeviceImage3D atten_map,
                                                   const Coordinate3D det_coordinate,
                                                   Value3<T> *pos_table_ptr, size_t num_rays_per_view,
                                                   Matrix4x4 *aff_list_ptr, size_t num_views,
                                                   T *projection_value)
            {
                int ind_ray = blockIdx.x * blockDim.x + threadIdx.x;
                int ind_view = blockIdx.y * blockDim.y + threadIdx.y;
                if (ind_ray < num_rays_per_view && ind_view < num_views)
                {
                    // get the thread index
                    int thread_idx = ind_ray + ind_view * num_rays_per_view;

                    // get the affine matrix
                    Matrix4x4 affine_view = aff_list_ptr[ind_view]; // TODO: use reference or copy?
                    auto det_translation = Matrix4x4::make_translation(pos_table_ptr[ind_ray]);
                    auto composed_affine = affine_view.compose(det_translation);

                    // update the projection value with distance-driven ray tracing
                    projection_value[thread_idx] = generic_project_kernel(image, atten_map, composed_affine, det_coordinate);
                }
            }

            /// @brief
            /// @param image
            /// @param det_coordinate
            /// @param pos_table_ptr
            /// @param trajectory_list
            /// @param projection_value
            void generic_project(Image3D &image,
                                 Image3D &atten_map,
                                 Coordinate3D &det_coordinate,
                                 Projection &projection)
            {
                typedef Matrix4x4 Affine;
                auto lookup_table = det_coordinate.make_lookup_table();
                Value3<T> *lookup_table_ptr = thrust::raw_pointer_cast(lookup_table.data());

                auto image_device = image.to_device();
                auto atten_map_device = atten_map.to_device();
                auto projection_device = projection.to_device();
                T *projection_value_ptr = projection_device.projection_value;

                auto trajectory_list = projection_device.trajectory_list;
                size_t ray_index = 0;

                int num_trajectories = trajectory_list.size();
                for (int i_traj = 0; i_traj < num_trajectories; i_traj++)
                {

                    auto current_traj = trajectory_list[i_traj];
                    auto affine_list = current_traj.get_affine_list();
                    Affine *affine_list_ptr = thrust::raw_pointer_cast(affine_list.data());

                    const size_t num_rays_per_view = det_coordinate.size.product();
                    const size_t num_views = current_traj.num_views;
                    const size_t total_num_rays = num_rays_per_view * num_views;

                    dim3 dimBlock(kThreadPerBlock_x, kThreadPerBlock_y);
                    dim3 dimGrid((num_rays_per_view + kThreadPerBlock_x - 1) / kThreadPerBlock_x,
                                 (num_views + kThreadPerBlock_y - 1) / kThreadPerBlock_y);
                    generic_project_global<<<dimGrid, dimBlock>>>(image_device, atten_map_device, det_coordinate,
                                                                  lookup_table_ptr, num_rays_per_view,
                                                                  affine_list_ptr, num_views,
                                                                  projection_value_ptr + ray_index);
                    ray_index += total_num_rays;
                }
                cudaDeviceSynchronize();
            }

            __device__ void generic_backproject_kernel(DeviceImage3D &image, DeviceImage3D &atten_map,
                                                       const Matrix4x4 &det_affine,
                                                       const Coordinate3D &det_coordinate,
                                                       T value)
            {
                auto ray_direction = det_affine.get_normal();
                auto image_coordinate = image.coordinate;
                auto image_size = image_coordinate.size;
                auto image_spacing = image_coordinate.spacing;

                auto det_size = det_coordinate.size;
                auto det_half_spacing = det_coordinate.spacing / 2;

                // const Value3<T> origin(0, 0, 0);

                Value3<T> det_left, det_right, det_bottom, det_top;
                set_detector_bounds(det_affine, det_coordinate, det_left, det_right, det_bottom, det_top);

                T atten_integral = 0.0;
                if (abs(ray_direction.x) > abs(ray_direction.y))
                {
                    // calculate the interaction of lor and image pixel
                    const T interaction_length = compute_interaction_ratio_on_x(ray_direction) * image_spacing.x;

                    // set the direction of ray to proceedding.
                    bool sign = ray_direction.x > 0 ? true : false;

                    // loop over x axis
                    for (size_t ix = 0; ix < image_size.x; ix++)
                    {
                        // if sign is negative, the ray is proceeding from right to left, so the index of image should be reoriented.
                        size_t reoriented_ix = sign ? ix : (image_size.x - ix - 1);
                        // calculat the current x center
                        const T pos_x = image.get_pos_x_by_index(reoriented_ix);

                        // ray do not go through the plane
                        if (ray_direction.x * (pos_x - det_left.x) < 0)
                            continue;
                        // TODO: time-of-flight weighting

                        // calculate the interaction point of lors on Y-Z plane
                        IndexInt bound_min, bound_max;
                        Value3<T> pos_det_min, pos_det_max;
                        compute_bounds_on_yz(image, det_left, det_right, det_bottom, det_top, ray_direction, pos_x, bound_min, bound_max, pos_det_min, pos_det_max);

                        // loop over yz plane to accumulate all the voxels
                        loop_yz_plane_bp(value, image, atten_map, atten_integral, interaction_length, bound_min, bound_max, pos_det_min, pos_det_max);
                    }
                }
                else // main axis is y
                {
                    const T interaction_length = compute_interaction_ratio_on_y(ray_direction) * image_spacing.y;

                    // set the direction of ray to proceedding.
                    bool sign = ray_direction.y > 0 ? true : false;

                    // loop over y axis
                    for (size_t iy = 0; iy < image_size.y; iy++)
                    {
                        // if sign is negative, the ray is proceeding from right to left, so the index of image should be reoriented.
                        size_t reoriented_iy = sign ? iy : (image_size.y - iy - 1);

                        const T pos_y = image.get_pos_y_by_index(reoriented_iy);
                        // ray do not go through the plane
                        if (ray_direction.y * (pos_y - det_left.y) < 0)
                            continue;

                        IndexInt bound_min, bound_max;
                        Value3<T> pos_det_min, pos_det_max;
                        compute_bounds_on_xz(image, det_left, det_right, det_bottom, det_top, ray_direction, pos_y, bound_min, bound_max, pos_det_min, pos_det_max);
                        loop_xz_plane_bp(value, image, atten_map, atten_integral, interaction_length, bound_min, bound_max, pos_det_min, pos_det_max);
                    }
                }
            }

            __global__ void generic_backproject_global(DeviceImage3D image, DeviceImage3D atten_map,
                                                       const Coordinate3D det_coordinate,
                                                       Value3<T> *pos_table_ptr, size_t num_rays_per_view,
                                                       Matrix4x4 *aff_list_ptr, size_t num_views,
                                                       T *projection_value)
            {
                int ind_ray = blockIdx.x * blockDim.x + threadIdx.x;
                int ind_view = blockIdx.y * blockDim.y + threadIdx.y;
                if (ind_ray < num_rays_per_view && ind_view < num_views)
                {
                    // get the thread index
                    int thread_idx = ind_ray + ind_view * num_rays_per_view;

                    // get the affine matrix
                    Matrix4x4 affine_view = aff_list_ptr[ind_view]; // TODO: use reference or copy?
                    auto det_translation = Matrix4x4::make_translation(pos_table_ptr[ind_ray]);
                    auto composed_affine = affine_view.compose(det_translation);

                    // update the projection value with distance-driven ray tracing
                    generic_backproject_kernel(image, atten_map, composed_affine, det_coordinate, projection_value[thread_idx]);
                }
            }

            void generic_backproject(Image3D &image,
                                     Image3D &atten_map,
                                     Coordinate3D &det_coordinate,
                                     Projection &projection)
            {
                typedef Matrix4x4 Affine;

                auto lookup_table = det_coordinate.make_lookup_table();
                Value3<T> *lookup_table_ptr = thrust::raw_pointer_cast(lookup_table.data());
                auto image_device = image.to_device();
                auto atten_map_device = atten_map.to_device();
                auto projection_device = projection.to_device();
                T *projection_value_ptr = projection_device.projection_value;
                auto trajectory_list = projection_device.trajectory_list;
                int num_trajectories = trajectory_list.size();
                size_t ray_index = 0;
                for (int i_traj = 0; i_traj < num_trajectories; i_traj++)
                {
                    auto current_traj = trajectory_list[i_traj];
                    auto affine_list = current_traj.get_affine_list();
                    Affine *affine_list_ptr = thrust::raw_pointer_cast(affine_list.data());
                    const size_t num_rays_per_view = det_coordinate.size.product();
                    const size_t num_views = current_traj.num_views;
                    const size_t total_num_rays = num_rays_per_view * num_views;
                    dim3 dimBlock(kThreadPerBlock_x, kThreadPerBlock_y);
                    dim3 dimGrid((num_rays_per_view + kThreadPerBlock_x - 1) / kThreadPerBlock_x,
                                 (num_views + kThreadPerBlock_y - 1) / kThreadPerBlock_y);
                    generic_backproject_global<<<dimGrid, dimBlock>>>(image_device, atten_map_device, det_coordinate,
                                                                      lookup_table_ptr, num_rays_per_view,
                                                                      affine_list_ptr, num_views,
                                                                      projection_value_ptr + ray_index);
                    ray_index += total_num_rays;
                }
            }

            __global__ void generic_cdr_project_global(const DeviceImage3D image,
                                                       const DeviceImage3D atten_map,
                                                       const Coordinate3D det_coordinate,
                                                       Value3<T> *pos_table_ptr, size_t num_rays_per_view,
                                                       Matrix4x4 *aff_list_ptr, size_t num_views,
                                                       const GaussianCDR cdr_model,
                                                       T *projection_value)
            {
                int ind_ray = blockIdx.x * blockDim.x + threadIdx.x;
                int ind_view = blockIdx.y * blockDim.y + threadIdx.y;
                if (ind_ray < num_rays_per_view && ind_view < num_views)
                {
                    // int thread_idx = ind_ray + ind_view * num_rays_per_view;

                    // get the affine matrix
                    Matrix4x4 affine_view = aff_list_ptr[ind_view];
                    auto det_translation = Matrix4x4::make_translation(pos_table_ptr[ind_ray]);
                    auto composed_affine = affine_view.compose(det_translation);

                    // project the image to the detector plane with cdr weighting
                    T value = cdr_project_kernel(image, atten_map, composed_affine, det_coordinate,
                                                 cdr_model, ind_ray, projection_value + ind_view * num_rays_per_view);
                }
            }

            __global__ void generic_cdr_backproject_global(DeviceImage3D image, DeviceImage3D atten_map,
                                                           const Coordinate3D det_coordinate,
                                                           Value3<T> *pos_table_ptr, size_t num_rays_per_view,
                                                           Matrix4x4 *aff_list_ptr, size_t num_views,
                                                           const GaussianCDR cdr_model,
                                                           T *projection_value)
            {
                int ind_ray = blockIdx.x * blockDim.x + threadIdx.x;
                int ind_view = blockIdx.y * blockDim.y + threadIdx.y;
                if (ind_ray < num_rays_per_view && ind_view < num_views)
                {
                    // int thread_idx = ind_ray + ind_view * num_rays_per_view;

                    // get the affine matrix
                    Matrix4x4 affine_view = aff_list_ptr[ind_view];
                    auto det_translation = Matrix4x4::make_translation(pos_table_ptr[ind_ray]);
                    auto composed_affine = affine_view.compose(det_translation);

                    // backproject the projection value to the image with cdr weighting
                    cdr_backproject_kernel(image, atten_map, composed_affine, det_coordinate,
                                           cdr_model, projection_value + ind_view * num_rays_per_view, ind_ray);
                }
            }

            void generic_cdr_project(Image3D &image,
                                     Image3D &atten_map,
                                     Coordinate3D &det_coordinate,
                                     Projection &projection,
                                     const GaussianCDR &cdr_model)
            {
                typedef Matrix4x4 Affine;

                auto lookup_table = det_coordinate.make_lookup_table();
                Value3<T> *lookup_table_ptr = thrust::raw_pointer_cast(lookup_table.data());
                auto image_device = image.to_device();
                auto atten_map_device = atten_map.to_device();
                auto projection_device = projection.to_device();
                T *projection_value_ptr = projection_device.projection_value;
                auto trajectory_list = projection_device.trajectory_list;
                size_t ray_index = 0;

                int num_trajectories = trajectory_list.size();
                for (int i_traj = 0; i_traj < num_trajectories; i_traj++)
                {
                    auto current_traj = trajectory_list[i_traj];
                    auto affine_list = current_traj.get_affine_list();
                    Affine *affine_list_ptr = thrust::raw_pointer_cast(affine_list.data());
                    const size_t num_rays_per_view = det_coordinate.size.product();
                    const size_t num_views = current_traj.num_views;
                    const size_t total_num_rays = num_rays_per_view * num_views;
                    dim3 dimBlock(kThreadPerBlock_x, kThreadPerBlock_y);
                    dim3 dimGrid((num_rays_per_view + kThreadPerBlock_x - 1) / kThreadPerBlock_x,
                                 (num_views + kThreadPerBlock_y - 1) / kThreadPerBlock_y);
                    generic_cdr_project_global<<<dimGrid, dimBlock>>>(image_device, atten_map_device, det_coordinate,
                                                                      lookup_table_ptr, num_rays_per_view,
                                                                      affine_list_ptr, num_views,
                                                                      cdr_model, projection_value_ptr + ray_index);
                    ray_index += total_num_rays;
                }
                cudaDeviceSynchronize();
            }

            void generic_cdr_backproject(Image3D &image,
                                         Image3D &atten_map,
                                         Coordinate3D &det_coordinate,
                                         Projection &projection,
                                         const GaussianCDR &cdr_model)
            {
                typedef Matrix4x4 Affine;

                auto lookup_table = det_coordinate.make_lookup_table();
                Value3<T> *lookup_table_ptr = thrust::raw_pointer_cast(lookup_table.data());
                auto image_device = image.to_device();
                auto atten_map_device = atten_map.to_device();
                auto projection_device = projection.to_device();
            
                T *projection_value_ptr = projection_device.projection_value;
                auto trajectory_list = projection_device.trajectory_list;

                size_t ray_index = 0;

                int num_trajectories = trajectory_list.size();
                for (int i_traj = 0; i_traj < num_trajectories; i_traj++)
                {
                    auto current_traj = trajectory_list[i_traj];
                    auto affine_list = current_traj.get_affine_list();
                    Affine *affine_list_ptr = thrust::raw_pointer_cast(affine_list.data());
                    const size_t num_rays_per_view = det_coordinate.size.product();
                    const size_t num_views = current_traj.num_views;
                    const size_t total_num_rays = num_rays_per_view * num_views;
                    dim3 dimBlock(kThreadPerBlock_x, kThreadPerBlock_y);
                    dim3 dimGrid((num_rays_per_view + kThreadPerBlock_x - 1) / kThreadPerBlock_x,
                                 (num_views + kThreadPerBlock_y - 1) / kThreadPerBlock_y);
                    generic_cdr_backproject_global<<<dimGrid, dimBlock>>>(image_device, atten_map_device, det_coordinate,
                                                                          lookup_table_ptr, num_rays_per_view,
                                                                          affine_list_ptr, num_views,
                                                                          cdr_model, projection_value_ptr + ray_index);
                    ray_index += total_num_rays;
                }
            }
        }
    }
}

NMRECON_NAMESPACE_END
