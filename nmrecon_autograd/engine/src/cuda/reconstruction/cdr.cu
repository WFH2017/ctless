#include "cuda/reconstruction/cdr.h"
#include <cmath>

NMRECON_NAMESPACE_BEGIN
namespace reconstruction
{
    namespace correction
    {
        namespace cdr
        {

            __device__ __host__
            GaussianCDR::GaussianCDR(const T &_a, const T &_b, const T &_t) : a(_a), b(_b), truncated_sigma_times(_t)
            {
            }

            __device__ __host__
            GaussianCDR::GaussianCDR(const GaussianCDR &other) : a(other.a), b(other.b),
                                                                 truncated_sigma_times(other.truncated_sigma_times)
            {
            }

            __device__ __host__ GaussianCDR &GaussianCDR::operator=(const GaussianCDR &other)
            {
                a = other.a;
                b = other.b;
                truncated_sigma_times = other.truncated_sigma_times;
                return *this;
            }

            __device__ __host__ size_t GaussianCDR::compute_valid_mapsize(const T &pixel_to_detector_distance,
                                                                          const T &detector_spacing) const
            {
                // compute fwhm and sigma using CDR model.
                T fwhm = a * pixel_to_detector_distance + b;
                T sigma = fwhm / k2_sqrt_Ln2;
                T truncated_sigma = sigma * truncated_sigma_times;
                // assume the detector spacing is the same in y and z direction
                // the map is a square and the size is odd.
                auto valid_mapsize = static_cast<size_t>(truncated_sigma / detector_spacing) * 2 + 1;

                // restrict the map size to be less than kMaxMapSize.
                if (valid_mapsize > kMaxMapSize)
                    valid_mapsize = kMaxMapSize;

                return valid_mapsize;
            }

            __device__ __host__ T GaussianCDR::compute_sum_weight(const T &detector_spacing, const T &sigma) const
            {
                T sum_weight = 0;
                T sigma_2 = sigma * sigma;

                for (int i_h = 0; i_h < kMaxMapSize; ++i_h)
                {
                    T dist_h = (i_h - kMaxMapSize / 2) * detector_spacing;
                    for (int i_w = 0; i_w < kMaxMapSize; ++i_w)
                    {
                        // compute the distance from the center of the map to the current pixel.
                        T dist_w = (i_w - kMaxMapSize / 2) * detector_spacing;
                        T dist_2 = dist_w * dist_w + dist_h * dist_h;
                        T weight = std::exp(-dist_2 / (2 * sigma_2)) / (2 * pi * sigma_2) * detector_spacing * detector_spacing;
                        // accumulate the value to the map.
                        sum_weight += weight;
                    }
                }
                T final_sum_weight = sum_weight * detector_spacing * detector_spacing;
                return final_sum_weight;
            }

            __device__ __host__ void GaussianCDR::compute_cdr_bounds(const Coordinate3D &det_coordinate,
                                                                     const int &ind_ray, const size_t &map_size,
                                                                     int &det_w_min, int &det_w_max,
                                                                     int &det_h_min, int &det_h_max,
                                                                     int &cdr_w_min, int &cdr_h_min) const
            {
                auto sw = det_coordinate.size.y;
                auto sh = det_coordinate.size.z;
                Index3D ind = det_coordinate.index1d_to_3d(ind_ray);
                int half_map_size = map_size / 2;
                det_w_min = max(0, static_cast<int>(ind.y) - half_map_size);
                det_w_max = min(static_cast<int>(sw) - 1, static_cast<int>(ind.y) + half_map_size);

                det_h_min = max(0, static_cast<int>(ind.z) - half_map_size);
                det_h_max = min(static_cast<int>(sh) - 1, static_cast<int>(ind.z) + half_map_size);
                cdr_w_min = half_map_size - (ind.y - det_w_min);
                cdr_h_min = half_map_size - (ind.z - det_h_min);
            }

            __device__ __host__ void GaussianCDR::project(const Coordinate3D &det_coordinate,
                                                          const T &pixel_to_detector_distance,
                                                          const int &ind_ray,
                                                          const T &image_value,
                                                          T *accumulated_cdr_values) const
            {
                int det_w_min, det_w_max, det_h_min, det_h_max;
                int cdr_h_min, cdr_w_min;
                compute_cdr_bounds(det_coordinate, ind_ray, kMaxMapSize, det_w_min, det_w_max, det_h_min, det_h_max, cdr_w_min, cdr_h_min);

                T sigma = (a * pixel_to_detector_distance + b) / k2_sqrt_Ln2;
                T sigma_2 = sigma * sigma;
                // Six-sigma tail cutoff: checked against original 49x49 operator.
                // Preserve original accumulator coordinates after narrowing detector bounds.
                const auto center = det_coordinate.index1d_to_3d(ind_ray);
                int radius_w = min(kMaxMapSize / 2, int(ceil(6.0f * abs(sigma) / det_coordinate.spacing.y)));
                int radius_h = min(kMaxMapSize / 2, int(ceil(6.0f * abs(sigma) / det_coordinate.spacing.z)));
                int new_w_min = max(det_w_min, int(center.y) - radius_w);
                int new_h_min = max(det_h_min, int(center.z) - radius_h);
                cdr_w_min += new_w_min - det_w_min;
                cdr_h_min += new_h_min - det_h_min;
                det_w_min = new_w_min; det_h_min = new_h_min;
                det_w_max = min(det_w_max, int(center.y) + radius_w);
                det_h_max = min(det_h_max, int(center.z) + radius_h);


                // compute the sum of the weight of the map to normalize the cdr map weight
                // no need to compute the sum weight becuase the weight is approximately normalized by multiplying the area of the pixel.
                // T sum_weight = compute_sum_weight(detector_spacing, sigma);

                // for each ray, loop the cdr map and compute the projection value
                for (int h = det_h_min; h <= det_h_max; h++)
                {
                    int cdr_h = h - det_h_min + cdr_h_min;
                    T dist_h = (cdr_h - kMaxMapSize / 2) * det_coordinate.spacing.z;

                    for (int w = det_w_min; w <= det_w_max; w++)
                    {
                        int cdr_w = w - det_w_min + cdr_w_min;
                        T dist_w = (cdr_w - kMaxMapSize / 2) * det_coordinate.spacing.y;
                        T dist_2 = dist_w * dist_w + dist_h * dist_h;
                        // compute the weight of the pixel.
                        T weight = std::exp(-dist_2 / (2 * sigma_2)) / (2 * pi * sigma_2) * det_coordinate.spacing.z * det_coordinate.spacing.y;

                        // int det_index = w + h * det_coordinate.size.y;
                        int cdr_index = (w - det_w_min + cdr_w_min) + (h - det_h_min + cdr_h_min) * kMaxMapSize;

                        // accumulate the projection value
                        accumulated_cdr_values[cdr_index] += weight * image_value;
                    }
                }
            }

            __device__ void GaussianCDR::update_projection(const Coordinate3D &det_coordinate,
                                                           const size_t &ind_ray,
                                                           const T *accumulated_cdr_values,
                                                           T *projection_value_current_view) const
            {
                // compute the bounds of the map projected on the detector.
                int det_w_min, det_w_max, det_h_min, det_h_max;
                int cdr_h_min, cdr_w_min;
                compute_cdr_bounds(det_coordinate, ind_ray, kMaxMapSize, det_w_min, det_w_max, det_h_min, det_h_max, cdr_w_min, cdr_h_min);

                // for each ray, loop the cdr map and compute the projection value
                for (int h = det_h_min; h <= det_h_max; h++)
                {
                    int cdr_h = h - det_h_min + cdr_h_min;
                    for (int w = det_w_min; w <= det_w_max; w++)
                    {
                        int cdr_w = w - det_w_min + cdr_w_min;
                        int det_index = w + h * det_coordinate.size.y;
                        int cdr_index = cdr_w + cdr_h * kMaxMapSize;

                        // compute the projection value
                        if (accumulated_cdr_values[cdr_index] != 0.0f) atomicAdd(projection_value_current_view + det_index, accumulated_cdr_values[cdr_index]);

                        // For DEBUG
                        // atomicAdd(projection_value_current_view + det_index, 1.0f);
                    }
                }
            }

            __device__ __host__ T GaussianCDR::convolve_projection(const Coordinate3D &det_coordinate,
                                                                   const T &pixel_to_detector_distance,
                                                                   const size_t &ind_ray,
                                                                   const T *projection_value_current_view) const
            {
                T result = 0.0;
                // compute the bounds of the map projected on the detector and corresponding bounds of cdr map.
                int det_w_min, det_w_max, det_h_min, det_h_max;
                int cdr_h_min, cdr_w_min;
                compute_cdr_bounds(det_coordinate, ind_ray, kMaxMapSize, det_w_min, det_w_max, det_h_min, det_h_max, cdr_w_min, cdr_h_min);

                T sigma = (a * pixel_to_detector_distance + b) / k2_sqrt_Ln2;
                T sigma_2 = sigma * sigma;
                // Six-sigma tail cutoff: checked against original 49x49 operator.
                // Preserve original accumulator coordinates after narrowing detector bounds.
                const auto center = det_coordinate.index1d_to_3d(ind_ray);
                int radius_w = min(kMaxMapSize / 2, int(ceil(6.0f * abs(sigma) / det_coordinate.spacing.y)));
                int radius_h = min(kMaxMapSize / 2, int(ceil(6.0f * abs(sigma) / det_coordinate.spacing.z)));
                int new_w_min = max(det_w_min, int(center.y) - radius_w);
                int new_h_min = max(det_h_min, int(center.z) - radius_h);
                cdr_w_min += new_w_min - det_w_min;
                cdr_h_min += new_h_min - det_h_min;
                det_w_min = new_w_min; det_h_min = new_h_min;
                det_w_max = min(det_w_max, int(center.y) + radius_w);
                det_h_max = min(det_h_max, int(center.z) + radius_h);


                // for each ray, loop the cdr map and compute the projection value
                for (int h = det_h_min; h <= det_h_max; h++)
                {
                    int cdr_h = h - det_h_min + cdr_h_min;
                    T dist_h = (cdr_h - kMaxMapSize / 2) * det_coordinate.spacing.z;
                    for (int w = det_w_min; w <= det_w_max; w++)
                    {
                        int cdr_w = w - det_w_min + cdr_w_min;
                        T dist_w = (cdr_w - kMaxMapSize / 2) * det_coordinate.spacing.y;
                        T dist_2 = dist_w * dist_w + dist_h * dist_h;
                        // compute the weight of the pixel.
                        T weight = std::exp(-dist_2 / (2 * sigma_2)) / (2 * pi * sigma_2) * det_coordinate.spacing.z * det_coordinate.spacing.y;

                        int det_index = w + h * det_coordinate.size.y;
                        // int cdr_index = (w - det_w_min + cdr_w_min) + (h - det_h_min + cdr_h_min) * kMaxMapSize;

                        // compute the projection value
                        result += *(projection_value_current_view + det_index) * weight;
                    }
                }
                return result;
            }

        } // namespace cdr
    } // namespace correction
} // namespace reconstruction
NMRECON_NAMESPACE_END
