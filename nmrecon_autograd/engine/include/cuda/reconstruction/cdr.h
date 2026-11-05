#pragma once

#include <cuda_runtime.h>
#include "common/config.h"
#include "scanner/spect.h"
#include "scanner/collimator.h"
#include "common/basic_type.h"
#include "cuda/image3d.h"


NMRECON_NAMESPACE_BEGIN
namespace reconstruction
{
  namespace correction
  {
    namespace cdr
    {

      // 2*sqrt(ln(2)) for the transform
      static constexpr precision_t k2_sqrt_Ln2 = 2.35482;
      static constexpr precision_t pi = 3.1415927;

      /**
       * @brief a fixed CDR kernel map size, the map size is odd number.
       * in the case of LEHR collimator, the simulated gaussain model is represented as
       * fwhm = 0.05357451 * pixel_to_detector_distance + 4.36830397 mm.
       * for a 3-sigma truncation, the map must cover 6 sigma in both width and height direction.
       * aussume the detector spacing is 2.5mm and maximum 800 mm detector to source distance,
       * the max map size is computed as (0.0535*800+4.368)/2.35482*6/2.5 = 48.07297,
       * considering the map size is odd number, we set max map size as 49.
       */
      static constexpr int kMaxMapSize = 49;

      using nmrecon::common::Coordinate3D;
      using nmrecon::common::DeviceImage3D;
      using nmrecon::common::Image3D;
      using nmrecon::common::Matrix4x4;
      using nmrecon::common::Value3;
      using nmrecon::scanner::SPECT;

      typedef Value3<size_t> Index3D;
      typedef Value3<int> IndexInt;
      typedef precision_t T;
      // const int kThreadPerBlock_x = 512;
      // const int kThreadPerBlock_y = 1;

      /**
       * @brief create a gaussian CDR model, the sigma of the CDR is calculated by
       * sigma = a * pixel_to_detector_distance + b.
       */
      struct GaussianCDR
      {

        T a;
        T b;
        T truncated_sigma_times;
        // 构造函数
        __device__ __host__ GaussianCDR(const T &_a, const T &_b, const T &_t);
        __device__ __host__ GaussianCDR(const GaussianCDR &other);
        __device__ __host__ GaussianCDR &operator=(const GaussianCDR &other);

        /**
         * @brief Compute the valid map size for the CDR model for a given pixel_to_detector_distance.
         * This is used to tracing an individual ray from the detector to the
         */
        __device__ __host__ size_t compute_valid_mapsize(const T &pixel_to_detector_distance,
                                                         const T &detector_spacing) const;

        /// @brief deprecated
        /// @param detector_spacing
        /// @param sigma
        /// @return
        __device__ __host__ T compute_sum_weight(const T &detector_spacing, const T &sigma) const;

        /// @brief compute the bounds of the ray on the detector plane.
        __device__ __host__ void compute_cdr_bounds(const Coordinate3D &det_coordinate,
                                                    const int &ind_ray, const size_t &map_size,
                                                    int &det_w_min, int &det_w_max,
                                                    int &det_h_min, int &det_h_max,
                                                    int &cdr_w_min, int &cdr_h_min) const;

        /// @brief map the image pixel value to the projection by the CDR model and accumulate.
        /// @param detector_spacing
        /// @param pixel_to_detector_distance
        /// @param image_value
        /// @param accumulated_cdr_values
        __device__ __host__ void project(const Coordinate3D &det_coordinate,
                                         const T &pixel_to_detector_distance,
                                         const int &ind_ray,
                                         const T &image_value,
                                         T *accumulated_cdr_values) const;

        // loop over the cdr map to update the overlapped projection value.
        __device__ void update_projection(const Coordinate3D &det_coordinate,
                                          const size_t &ind_ray,
                                          const T *accumulated_cdr_values,
                                          T *projection_value_current_view) const;

        /// @brief convolve the projection value with the cdr model for backprojection.
        /// @param det_coordinate
        __device__ __host__ T convolve_projection(const Coordinate3D &det_coordinate,
                                                  const T &pixel_to_detector_distance,
                                                  const size_t &ind_ray,
                                                  const T *projection_value_current_view) const;
      };
    } // namespace cdr
  } // namespace correction
} // namespace reconstruction
NMRECON_NAMESPACE_END