#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "common/config.h"
#include "scanner/spect.h"
#include "scanner/collimator.h"
#include "common/basic_type.h"
#include "cuda/image3d.h"
#include "cuda/reconstruction/cdr.h"
#include "cuda/reconstruction/projection.h"

NMRECON_NAMESPACE_BEGIN
namespace reconstruction
{
    namespace projector
    {
        namespace distance_driven
        {


            typedef nmrecon::common::Value3<size_t> Size;
            typedef nmrecon::common::Value3<size_t> Index;
            typedef nmrecon::common::Value3<int> IndexInt;
            typedef thrust::device_vector<precision_t> ProjectionValue;
            typedef precision_t T;

            using nmrecon::common::Coordinate3D;
            using nmrecon::common::DeviceImage3D;
            using nmrecon::common::Image3D;
            using nmrecon::common::Matrix4x4;
            using nmrecon::common::Value3;
            using nmrecon::scanner::SPECT;
            using nmrecon::scanner::TrajectoryList;
            using nmrecon::reconstruction::rawdata::Projection;
            using nmrecon::reconstruction::rawdata::DeviceProjection;

            using namespace nmrecon::reconstruction::correction::cdr;
            __device__ inline void _sort(T &a, T &b)
            {
                if (a > b)
                {
                    T tmp = a;
                    a = b;
                    b = tmp;
                }
            }

            __device__ inline T safe_inv_cos(const T &dy, const T &dx)
            {
                if (std::abs(dx) < nmrecon::common::kEpsilon)
                {
                    return T(1.0);
                }
                const T tan_value = dy / dx;
                return sqrtf(1 + tan_value * tan_value);
            }

            /**
             * @brief calculate the interaction point of a line(point2-point1) on y axis of the plane x
             *
             * @tparam T
             * @param point1
             * @param point2
             * @param plane_x
             * @return __device__
             */
            __device__ inline T interact_y_on_plane_x(const Value3<T> point, const Value3<T> ray_direction, const T &plane_x)
            {
                return (plane_x - point.x) / ray_direction.x * ray_direction.y + point.y;
            }

            /**
             * @brief calculate the interaction point of a line(point2-point1) on z axis of the plane x
             *
             * @tparam T
             * @param point1
             * @param point2
             * @param plane_x
             * @return __device__
             */

            __device__ inline T interact_z_on_plane_x(const Value3<T> point, const Value3<T> ray_direction, const T &plane_x)
            {

                return (plane_x - point.x) / ray_direction.x * ray_direction.z + point.z;
            }

            /**
             * @brief calculate the interaction point of a line(point2-point1) on x axis of the plane y
             *
             * @tparam T
             * @param point1
             * @param point2
             * @param plane_y
             * @return __device__
             */

            __device__ inline T interact_x_on_plane_y(const Value3<T> point, const Value3<T> ray_direction, const T &plane_y)
            {
                return (plane_y - point.y) / ray_direction.y * ray_direction.x + point.x;
            }

            /**
             * @brief calculate the interaction point of a line(point2-point1) on z axis of the plane y
             *
             * @tparam T
             * @param point1
             * @param point2
             * @param plane_y
             * @return __device__
             */

            __device__ inline T interact_z_on_plane_y(const Value3<T> point, const Value3<T> ray_direction, const T &plane_y)
            {

                return (plane_y - point.y) / ray_direction.y * ray_direction.z + point.z;
            }

            __device__ inline T compute_interaction_ratio_on_x(const Value3<T> &ray_dir)
            {
                // compute the length of interaction of ray and image pixel in unit spacing.
                const T inv_cos_alpha = safe_inv_cos(ray_dir.y, ray_dir.x);
                const T inv_cos_gamma = safe_inv_cos(ray_dir.z, ray_dir.x);
                T interaction_ratio = inv_cos_alpha * inv_cos_gamma;
                return interaction_ratio;
            }

            __device__ inline T compute_interaction_ratio_on_y(const Value3<T> &ray_dir)
            {
                // compute the length of interaction of ray and image pixel in unit spacing.
                const T inv_cos_alpha = safe_inv_cos(ray_dir.x, ray_dir.y);
                const T inv_cos_gamma = safe_inv_cos(ray_dir.z, ray_dir.y);
                T interaction_ratio = inv_cos_alpha * inv_cos_gamma;
                return interaction_ratio;
            }

            /**
             * @brief for projection without atteuation
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
                                       const Value3<T> &pos1_det, const Value3<T> &pos2_det);

            /**
             * @brief for projection with attenuation
             *
             * @tparam T
             * @param image
             * @param atten_map
             * @param atten_integral
             * @param interaction_length
             * @param bound_min
             * @param bound_max
             * @param pos1_det
             * @param pos2_det
             * @return __device__
             */

            __device__ T loop_yz_plane(const DeviceImage3D &image,
                                       const DeviceImage3D &atten_map, T &atten_integral,
                                       const T &interaction_length,
                                       const IndexInt &bound_min, const IndexInt &bound_max,
                                       const Value3<T> &pos1_det, const Value3<T> &pos2_det);

            /**
             * @brief for backprojection without attenuation
             *
             * @tparam T
             * @param value
             * @param image
             * @param bound_min
             * @param bound_max
             * @param pos1_det
             * @param pos2_det
             * @return __device__
             */
            __device__ void loop_yz_plane_bp(const T &value, DeviceImage3D &image,
                                             const IndexInt &bound_min, const IndexInt &bound_max,
                                             const Value3<T> &pos1_det, const Value3<T> &pos2_det);

            /**
             * @brief for backprojection with attenuation
             *
             * @tparam T
             * @param image
             * @param atten_map
             * @param atten_integral
             * @param interaction_length
             * @param bound_min
             * @param bound_max
             * @param pos1_det
             * @param pos2_det
             * @return void
             */
            __device__ void loop_yz_plane_bp(const T &value, DeviceImage3D &image,
                                              const DeviceImage3D &atten_map, T &atten_integral,
                                              const T &interaction_length,
                                              const IndexInt &bound_min, const IndexInt &bound_max,
                                              const Value3<T> &pos1_det, const Value3<T> &pos2_det);

            /**
             * @brief for projection without attenuation correction.
             *
             * @tparam T
             * @param image
             * @param bound_min
             * @param bound_max
             * @param pos1_det
             * @param pos2_det
             * @return __device__
             */
            __device__ T loop_xz_plane(const DeviceImage3D &image,
                                       const IndexInt &bound_min, const IndexInt &bound_max,
                                       const Value3<T> &pos1_det, const Value3<T> &pos2_det);

            /**
             * @brief for projection with attenuation correction.
             *
             * @tparam T
             * @param image
             * @param atten_map
             * @param atten_integral
             * @param interaction_length
             * @param bound_min
             * @param bound_max
             * @param pos1_det
             * @param pos2_det
             * @return __device__
             */
            __device__ T loop_xz_plane(const DeviceImage3D &image,
                                       const DeviceImage3D &atten_map, T &atten_integral,
                                       const T &interaction_length,
                                       const IndexInt &bound_min, const IndexInt &bound_max,
                                       const Value3<T> &pos1_det, const Value3<T> &pos2_det);

            /**
             * @brief for backprojection without attenuation correction.
             *
             * @tparam T
             * @param image
             * @param bound_min
             * @param bound_max
             * @param pos1_det
             * @param pos2_det
             * @return __device__
             */
            __device__ void loop_xz_plane_bp(const T &value, DeviceImage3D &image,
                                             const IndexInt &bound_min, const IndexInt &bound_max,
                                             const Value3<T> &pos1_det, const Value3<T> &pos2_det);

            /**
             * @brief for backprojection with attenuation correction.
             *
             * @tparam T
             * @param image
             * @param atten_map
             * @param atten_integral
             * @param interaction_length
             * @param bound_min
             * @param bound_max
             * @param pos1_det
             * @param pos2_det
             * @return __device__
             */
            __device__ void loop_xz_plane_bp(const T &value, DeviceImage3D &image,
                                              const DeviceImage3D &atten_map, T &atten_integral,
                                              const T &interaction_length,
                                              const IndexInt &bound_min, const IndexInt &bound_max,
                                              const Value3<T> &pos1_det, const Value3<T> &pos2_det);

            /// @brief 已知探测器的位置和方向，计算探测器在y轴和z轴上的边界
            __device__ void set_detector_bounds(const Matrix4x4 &det_affine,
                                                const Coordinate3D &det_coordinate,
                                                Value3<T> &det_left, Value3<T> &det_right,
                                                Value3<T> &det_bottom, Value3<T> &det_top);

            /**
             * @brief compute the bounds of the ray on the image x-z plane,
             *        the bounds are used to loop the plane to compute the projection or backprojection values.
             * @tparam T
             * @param image
             * @param det_left
             * @param det_right
             * @param det_bottom
             * @param det_top
             * @param rtion
             * @param pos_y
             * @param bound_min : the minimum index of the ray on the image xay_direc-z plane.
             * @param bound_max : the maximum index of the ray on the image x-z plane.
             * @param pos_det_min : the minimum position of the ray on the image x-z plane.
             * @param pos_det_max : the maximum position of the ray on the image x-z plane.
             * @return __device__
             */
            __device__ void compute_bounds_on_xz(const DeviceImage3D &image,
                                                 const Value3<T> &det_left, const Value3<T> &det_right,
                                                 const Value3<T> &det_bottom, const Value3<T> &det_top,
                                                 const Value3<T> &ray_direction, const T &pos_y,
                                                 IndexInt &bound_min, IndexInt &bound_max,
                                                 Value3<T> &pos_det_min, Value3<T> &pos_det_max);

            /**
             * @brief compute the bounds of the ray on the image y-z plane,
             *        the bounds are used to loop the plane to compute the projection or backprojection values.
             * @tparam T
             * @param image
             * @param det_left : the left position of the detector.
             * @param det_right : the right position of the detector.
             * @param det_bottom : the bottom position of the detector.
             * @param det_top : the top position of the detector.
             * @param ray_direction : the direction of the ray (start from the detector and).
             * @param pos_x : the x position of the ray on the y-z plane.
             * @param bound_min : the minimum index of the interacted pixel on the image y-z plane.
             * @param bound_max : the maximum index of the interacted pixel on the image y-z plane.
             * @param pos_det_min : the minimum position of the ray on the image y-z plane.
             * @param pos_det_max : the maximum position of the ray on the image y-z plane.
             * @return __device__
             */
            __device__ void compute_bounds_on_yz(const DeviceImage3D &image,
                                                 const Value3<T> &det_left, const Value3<T> &det_right,
                                                 const Value3<T> &det_bottom, const Value3<T> &det_top,
                                                 const Value3<T> &ray_direction, const T &pos_x,
                                                 IndexInt &bound_min, IndexInt &bound_max,
                                                 Value3<T> &pos_det_min, Value3<T> &pos_det_max);

            __device__ T project_kernel(const DeviceImage3D &image,
                                        const Matrix4x4 &det_affine,
                                        const Coordinate3D &det_coordinate);

            static __global__ void project_global(const DeviceImage3D image,
                                                  Value3<T> *pos_table_ptr, size_t num_rays_per_view,
                                                  Matrix4x4 *aff_list_ptr, size_t num_views,
                                                  const Coordinate3D det_coordinate, T *projection_value);

            /**
             * @brief project the image to the detector
             * TODO: the image should be const, while it leads to error at the moment.
             * @tparam T
             * @param scanner
             * @param image
             * @param projection_value
             */
            void project(const SPECT &scanner, Image3D &image, ProjectionValue &projection_value);

            __device__ T atten_project_kernel(const DeviceImage3D &image,
                                              const DeviceImage3D &atten_map,
                                              const Matrix4x4 &det_affine,
                                              const Coordinate3D &det_coordinate);

            static __global__ void atten_project_global(const DeviceImage3D image,
                                                        const DeviceImage3D atten_map,
                                                        Value3<T> *pos_table_ptr, size_t num_rays_per_view,
                                                        Matrix4x4 *aff_list_ptr, size_t num_views,
                                                        const Coordinate3D det_coordinate, T *projection_value);

            /// @brief project the attenuation map to the detector along each ray and get the projection value.
            /// @tparam T
            /// @tparam ProjectionValue
            /// @param scanner
            /// @param image
            /// @param atten_map
            /// @param projection_value
            void atten_project(const SPECT &scanner, Image3D &image,
                               Image3D &atten_map, ProjectionValue &projection_value);

            __device__ void backproject_kernel(DeviceImage3D &image,
                                               const Matrix4x4 &det_affine,
                                               const Coordinate3D &det_coordinate,
                                               T value);

            __global__ void backproject_global(DeviceImage3D image,
                                               Value3<T> *pos_table_ptr, size_t num_rays_per_view,
                                               Matrix4x4 *aff_list_ptr, size_t num_views,
                                               const Coordinate3D det_coordinate, T *projection_value);

            void backproject(const SPECT &scanner, Image3D &image, ProjectionValue &projection_value);

            __device__ void atten_backproject_kernel(DeviceImage3D &image, DeviceImage3D &atten_map,
                                                     const Matrix4x4 &det_affine,
                                                     const Coordinate3D &det_coordinate,
                                                     T value);

            static __global__ void atten_backproject_global(DeviceImage3D image, DeviceImage3D atten_map,
                                                            Value3<T> *pos_table_ptr, size_t num_rays_per_view,
                                                            Matrix4x4 *aff_list_ptr, size_t num_views,
                                                            const Coordinate3D det_coordinate, T *projection_value);

            /// @brief backproject with attenuation correction
            /// @tparam T
            /// @param scanner
            /// @param image
            /// @param atten_map
            /// @param projection_value

            void atten_backproject(const SPECT &scanner,
                                   Image3D &image,
                                   Image3D &atten_map,
                                   ProjectionValue &projection_value);

            /// @brief project the image to the detector with cdr model
            /// @details the cdr model is used to accumulate the value of the image pixel,
            /// 射线从探测器到图像，对每个图像像素进行积分，积分值由CDR模型进行累加。
            ///
            /// @tparam T
            /// @param image
            /// @param atten_map
            /// @param cdr_model
            /// @param det_affine
            /// @param det_coordinate
            /// @param accumulated_cdr_values
            /// @return

            __device__ T cdr_project_kernel(const DeviceImage3D &image,
                                            const DeviceImage3D &atten_map,
                                            const Matrix4x4 &det_affine,
                                            const Coordinate3D &det_coordinate,
                                            const GaussianCDR &cdr_model,
                                            const int &ind_ray,
                                            T *projection_value_current_view);

            static __global__ void cdr_project_global(const DeviceImage3D image,
                                                      const DeviceImage3D atten_map,
                                                      Value3<T> *pos_table_ptr, size_t num_rays_per_view,
                                                      Matrix4x4 *aff_list_ptr, size_t num_views,
                                                      const Coordinate3D det_coordinate,
                                                      const GaussianCDR cdr_model,
                                                      T *projection_value);

            void cdr_project(const SPECT &scanner,
                             const GaussianCDR &cdr_model,
                             Image3D &image,
                             Image3D &atten_map,
                             ProjectionValue &projection_value);

            /// @brief backproject the projection value to the image with cdr model
            /// @tparam T
            /// @param image
            /// @param atten_map
            /// @param det_affine
            /// @param det_coordinate
            /// @param projection_value
            /// @param cdr_model
            /// @param ind_ray
            /// @return
            __device__ void cdr_backproject_kernel(DeviceImage3D &image,
                                                   DeviceImage3D &atten_map,
                                                   const Matrix4x4 &det_affine,
                                                   const Coordinate3D &det_coordinate,
                                                   const GaussianCDR &cdr_model,
                                                   const T *projection_value_current_view,
                                                   const int ind_ray);

            static __global__ void cdr_backproject_global(DeviceImage3D image, DeviceImage3D atten_map,
                                                          Value3<T> *pos_table_ptr, size_t num_rays_per_view,
                                                          Matrix4x4 *aff_list_ptr, size_t num_views,
                                                          const Coordinate3D det_coordinate,
                                                          const GaussianCDR cdr_model, T *projection_value);

            void cdr_backproject(const SPECT &scanner,
                                 const GaussianCDR &cdr_model,
                                 Image3D &image,
                                 Image3D &atten_map,
                                 ProjectionValue &projection_value);

            void generic_project(Image3D &image,
                                 Image3D &atten_map,
                                 Coordinate3D &det_coordinate,
                                 Projection &projection);

            void generic_backproject(Image3D &image,
                                     Image3D &atten_map,
                                     Coordinate3D &det_coordinate,
                                     Projection &projection);
            
            void generic_cdr_project(Image3D &image,
                                     Image3D &atten_map,
                                     Coordinate3D &det_coordinate,
                                     Projection &projection,
                                     const GaussianCDR &cdr_model);
                                     
            void generic_cdr_backproject(Image3D &image,
                                         Image3D &atten_map,
                                         Coordinate3D &det_coordinate,
                                         Projection &projection,
                                         const GaussianCDR &cdr_model);

        }
    }
}

NMRECON_NAMESPACE_END