
#pragma once

#include <cuda_runtime.h>
#include <algorithm> // for std::max on host side
#include "common/types.h"

NMRECON_NAMESPACE_BEGIN

namespace common
{
  static constexpr precision_t kEpsilon = 1e-6f;
  static constexpr precision_t kPi = 3.14159265358979323846f;


  /**
   * @brief Value3 contains three values of type T, which can be used to represent a 3D point
   * , a size, an orientation, etc.
   * @tparam T
   */
  template <typename T = precision_t>
  struct Value3
  {
    T x;
    T y;
    T z;

    __device__ __host__ Value3(const Value3 &vec);

    __device__ __host__ Value3(const T &x_ = 0, const T &y_ = 0, const T &z_ = 0);

    __device__ __host__ Value3(const T *data);
    // __device__
    // Value3(const OtherVec3& vec) : x(T(vec.x)), y(T(vec.y)), z(T(vec.z)) {}

    __device__ __host__ Value3 revert() const;

    /**
     * @brief add another Value3 to this Value3
     *
     * @param v
     * @return Value3
     */
    __device__ __host__ Value3 operator+(const Value3 &v) const;

    /**
     * @brief subtract another Value3 from this Value3
     *
     * @param v
     * @return Value3
     */
    __device__ __host__ Value3 operator-(const Value3 &v) const;

    /**
     * @brief multiply this Value3 by another Value3
     *
     * @param v
     * @return Value3
     */
    __device__ __host__ Value3 operator*(const Value3 &v) const;
    /**
     * @brief divide this Value3 by another Value3
     *
     * @param v
     * @return Value3
     */
    __device__ __host__ Value3 operator/(const Value3 &v) const;
    /**
     * @brief add another Value3 by a scalar
     *
     * @param v
     * @return Value3
     */
    __device__ __host__ Value3 operator+(const T &s) const;

    /**
     * @brief subtract another Value3 by a scalar
     *
     * @param v
     * @return Value3
     */
    __device__ __host__ Value3 operator-(const T &s) const;

    /**
     * @brief multiply this Value3 by a scalar
     *
     * @param s
     * @return Value3
     */
    __device__ __host__ Value3 operator*(const T &s) const;

    /**
     * @brief divide this Value3 by a scalar
     *
     * @param s
     * @return Value3
     */
    __device__ __host__ Value3 operator/(const T &s) const;

    /**
     * @brief return the bigger value of x and y
     *
     * @return T
     */
    __device__ __host__ T big_axis() const;

    /**
     * @brief return the smaller value of x and y
     *
     * @return T
     */
    __device__ __host__ T small_axis() const;

    __device__ __host__ size_t big_axis_index();

    __device__ __host__ size_t small_axis_index();
    /**
     * @brief return the length of this Value3
     *
     * @return T
     */
    __device__ __host__ T length() const;
    /**
     * @brief return the production of all elements
     *
     */
    __device__ __host__ T product() const;

    /**
     * @brief return sum of all elements
     *
     */
    __device__ __host__ T sum() const;

    __device__ __host__ Value3 norm() const;
    /**
     * @brief print this Value3
     *
     */
    __device__ __host__ void print() const;
  };

  typedef Value3<precision_t> Origin;
  typedef Value3<size_t> Size;
  typedef Value3<precision_t> Spacing;
  typedef Value3<precision_t> Vec3;

  struct Matrix3x3
  {
    precision_t m11, m12, m13;
    precision_t m21, m22, m23;
    precision_t m31, m32, m33;

    __device__ __host__
    Matrix3x3(const precision_t &m11_ = 1, const precision_t &m12_ = 0, const precision_t &m13_ = 0,
              const precision_t &m21_ = 0, const precision_t &m22_ = 1, const precision_t &m23_ = 0,
              const precision_t &m31_ = 0, const precision_t &m32_ = 0, const precision_t &m33_ = 1);

    __device__ __host__ Matrix3x3 compose(const Matrix3x3 &mat) const;

    __device__ __host__ Vec3 transform(const Vec3 &vec) const;

    __device__ __host__ bool is_identity() const;

    __device__ __host__ Vec3 get_normal() const;

    __device__ __host__ void print() const;

    static __device__ __host__ Matrix3x3 make_identity();
    static __device__ __host__ Matrix3x3 make_scale(const precision_t &scale1, const precision_t &scale2, const precision_t &scale3);
    static __device__ __host__ Matrix3x3 make_rotation_z(const precision_t &theta);
  };

  struct Matrix4x4
  {
    typedef Matrix3x3 Mat3x3;

    precision_t m11, m12, m13, m14;
    precision_t m21, m22, m23, m24;
    precision_t m31, m32, m33, m34;
    precision_t m41, m42, m43, m44;

    __device__ __host__
    Matrix4x4(const precision_t &m11_ = 1, const precision_t &m12_ = 0, const precision_t &m13_ = 0, const precision_t &m14_ = 0,
              const precision_t &m21_ = 0, const precision_t &m22_ = 1, const precision_t &m23_ = 0, const precision_t &m24_ = 0,
              const precision_t &m31_ = 0, const precision_t &m32_ = 0, const precision_t &m33_ = 1, const precision_t &m34_ = 0,
              const precision_t &m41_ = 0, const precision_t &m42_ = 0, const precision_t &m43_ = 0, const precision_t &m44_ = 1);

    __device__ __host__ Matrix4x4 compose(const Matrix4x4 &mat) const;

    __device__ __host__ Matrix4x4 from_Matrix3x3(const Mat3x3 &mat);
    __device__ __host__ Mat3x3 to_Matrix3x3() const;

    __device__ __host__ Vec3 transform(const Vec3 &vec) const;

    /**
     * @brief Get the normal vector of the transformed vector
     * used to compute the ray direction of the detector element,
     * thus (-1, 0, 0) is the original normal is the opposite direction from the detector.
     * @return
     */
    __device__ __host__ Vec3 get_normal() const;
    __device__ __host__ bool is_identity() const;

    __device__ __host__ void print() const;

    static __device__ __host__ Matrix4x4 make_identity();
    static __device__ __host__ Matrix4x4 make_scale(const Vec3 &vec);
    static __device__ __host__ Matrix4x4 make_scale(const precision_t &scale1, const precision_t &scale2, const precision_t &scale3);
    static __device__ __host__ Matrix4x4 make_translation(const Vec3 &vec);
    static __device__ __host__ Matrix4x4 make_translation(const precision_t &x, const precision_t &y, const precision_t &z);

    static __device__ __host__ Matrix4x4 make_rotation_z(const precision_t &theta);
  };

  /**
   * @brief thrust method to generate lookup table for cartesian grid.
   *
   * @tparam T
   */

  struct lookup_table_functor
  {
    const Origin origin;
    const Size size;
    const Spacing spacing;
    const Matrix3x3 direction;

    lookup_table_functor(const Origin &origin_,
                         const Size &size_,
                         const Spacing &spacing_,
                         const Matrix3x3 &direction_)
        : origin(origin_), size(size_), spacing(spacing_), direction(direction_) {}

    __device__ __host__ Vec3 operator()(const size_t &index) const;
  };

  typedef Matrix3x3 Direction;
  typedef thrust::device_vector<Vec3> Point3DVector;
  typedef thrust::host_vector<size_t> IndexVector;
  struct Coordinate3D
  {
    /**
     * @brief center of the first voxel
     *
     */
    Origin origin;

    /**
     * @brief number of voxels in each dimension
     *
     */
    Size size;

    /**
     * @brief voxel size in each dimension
     *
     */
    Spacing spacing;

    /**
     * @brief orientation of coordinate system
     *
     */
    Direction direction;

    __device__ __host__
    Coordinate3D(const Origin &origin_ = Origin(0.0, 0.0, 0.0), const Size &size_ = Size(1, 1, 1),
                 const Spacing &spacing_ = Spacing(1.0, 1.0, 1.0), const Direction &direction_ = Direction::make_identity())
        : origin(origin_), size(size_), spacing(spacing_), direction(direction_) {}

    __device__ __host__ Coordinate3D(const Coordinate3D &coord)
        : origin(coord.origin), size(coord.size), spacing(coord.spacing), direction(coord.direction) {}

    __device__ __host__ Coordinate3D &operator=(const Coordinate3D &coord)
    {
      origin = coord.origin;
      size = coord.size;
      spacing = coord.spacing;
      direction = coord.direction;
      return *this;
    }

    __device__ __host__ void set_origin(const Origin &origin_)
    {
      origin = origin_;
    }

    __device__ __host__ void set_size(const Size &size_)
    {
      size = size_;
    }

    __device__ __host__ void set_spacing(const Spacing &spacing_)
    {
      spacing = spacing_;
    }

    __device__ __host__ void set_direction(const Direction &direction_)
    {
      direction = direction_;
    }

    __device__ __host__ Vec3 get_physic_extent() const
    {
      return spacing * Vec3(size.x, size.y, size.z);
    }

    __device__ __host__ Vec3 get_physic_center() const
    {
      return origin + direction.transform((get_physic_extent() - spacing) / 2.0);
    }

    __device__ __host__ Vec3 get_pos_by_index(const size_t &index) const
    {
      printf("line %d, file %s\n", __LINE__, __FILE__);
      printf("not implemented yet\n");
      return origin;
    }

    __device__ __host__ Size index1d_to_3d(const size_t &index) const
    {

      const size_t ind_x = index % size.x;
      const size_t ind_y = (index / size.x) % size.y;
      const size_t ind_z = index / (size.x * size.y);
      return Size(ind_x, ind_y, ind_z);
    }

    __host__ Point3DVector make_lookup_table() const;
    __device__ __host__ void print() const;
    
  };

  typedef thrust::host_vector<precision_t> HostVector;
  typedef thrust::device_vector<precision_t> DeviceVector;
} // namespace common

NMRECON_NAMESPACE_END
