#include "common/basic_type.h"
#include <iostream> // for std::cout
#include <math.h>

NMRECON_NAMESPACE_BEGIN

namespace common
{

    // Value3 类的定义
    template <typename T>
    __device__ __host__ Value3<T>::Value3(const Value3 &vec) : x(vec.x), y(vec.y), z(vec.z) {}

    template <typename T>
    __device__ __host__ Value3<T>::Value3(const T &x_, const T &y_, const T &z_) : x(x_), y(y_), z(z_) {}

    template <typename T>
    __device__ __host__ Value3<T>::Value3(const T *data)
    {
        x = data[0];
        y = data[1];
        z = data[2];
    }

    template <typename T>
    __device__ __host__ Value3<T> Value3<T>::revert() const
    {
        return Value3(-x, -y, -z);
    }

    template <typename T>
    __device__ __host__ Value3<T> Value3<T>::operator+(const Value3 &v) const
    {
        return Value3(x + v.x, y + v.y, z + v.z);
    }

    template <typename T>
    __device__ __host__ Value3<T> Value3<T>::operator-(const Value3 &v) const
    {
        return Value3(x - v.x, y - v.y, z - v.z);
    }

    template <typename T>
    __device__ __host__ Value3<T> Value3<T>::operator*(const Value3 &v) const
    {
        return Value3(x * v.x, y * v.y, z * v.z);
    }

    template <typename T>
    __device__ __host__ Value3<T> Value3<T>::operator/(const Value3 &v) const
    {
        return Value3(x / v.x, y / v.y, z / v.z);
    }

    template <typename T>
    __device__ __host__ Value3<T> Value3<T>::operator+(const T &s) const
    {
        return Value3(x + s, y + s, z + s);
    }

    template <typename T>
    __device__ __host__ Value3<T> Value3<T>::operator-(const T &s) const
    {
        return Value3(x - s, y - s, z - s);
    }

    template <typename T>
    __device__ __host__ Value3<T> Value3<T>::operator*(const T &s) const
    {
        return Value3(x * s, y * s, z * s);
    }

    template <typename T>
    __device__ __host__ Value3<T> Value3<T>::operator/(const T &s) const
    {
        return Value3(x / s, y / s, z / s);
    }

    template <typename T>
    __device__ __host__ T Value3<T>::big_axis() const
    {
        return max(x, y);
    }

    template <typename T>
    __device__ __host__ T Value3<T>::small_axis() const
    {
        return min(x, y);
    }

    template <typename T>
    __device__ __host__ size_t Value3<T>::big_axis_index()
    {
        if (x > y)
            return 0;
        else
            return 1;
    }

    template <typename T>
    __device__ __host__ size_t Value3<T>::small_axis_index()
    {
        if (x < y)
            return 0;
        else
            return 1;
    }

    template <typename T>
    __device__ __host__ T Value3<T>::length() const
    {
        return sqrtf(x * x + y * y + z * z);
    }

    template <typename T>
    __device__ __host__ T Value3<T>::product() const
    {
        return x * y * z;
    }

    template <typename T>
    __device__ __host__ T Value3<T>::sum() const
    {
        return x + y + z;
    }

    template <typename T>
    __device__ __host__ Value3<T> Value3<T>::norm() const
    {
        T len = length();
        return (len > kEpsilon) ? Value3(x / len, y / len, z / len) : Value3(0, 0, 0);
    }

    template <typename T>
    __device__ __host__ void Value3<T>::print() const
    {
        printf("[ %f, %f, %f ] \n", float(x), float(y), float(z));
    }

    // Matrix3x3 类的实现
    __device__ __host__ Matrix3x3::Matrix3x3(const precision_t &m11_, const precision_t &m12_, const precision_t &m13_,
                                             const precision_t &m21_, const precision_t &m22_, const precision_t &m23_,
                                             const precision_t &m31_, const precision_t &m32_, const precision_t &m33_)
        : m11(m11_), m12(m12_), m13(m13_), m21(m21_), m22(m22_), m23(m23_), m31(m31_), m32(m32_), m33(m33_)
    {
    }

    __device__ __host__ Matrix3x3 Matrix3x3::compose(const Matrix3x3 &mat) const
    {
        return Matrix3x3(
            m11 * mat.m11 + m12 * mat.m21 + m13 * mat.m31,
            m11 * mat.m12 + m12 * mat.m22 + m13 * mat.m32,
            m11 * mat.m13 + m12 * mat.m23 + m13 * mat.m33,
            m21 * mat.m11 + m22 * mat.m21 + m23 * mat.m31,
            m21 * mat.m12 + m22 * mat.m22 + m23 * mat.m32,
            m21 * mat.m13 + m22 * mat.m23 + m23 * mat.m33,
            m31 * mat.m11 + m32 * mat.m21 + m33 * mat.m31,
            m31 * mat.m12 + m32 * mat.m22 + m33 * mat.m32,
            m31 * mat.m13 + m32 * mat.m23 + m33 * mat.m33);
    }

    __device__ __host__ Vec3 Matrix3x3::transform(const Vec3 &vec) const
    {
        return Vec3(
            m11 * vec.x + m12 * vec.y + m13 * vec.z,
            m21 * vec.x + m22 * vec.y + m23 * vec.z,
            m31 * vec.x + m32 * vec.y + m33 * vec.z);
    }

    __device__ __host__ bool Matrix3x3::is_identity() const
    {
        return abs(m11 - 1) < kEpsilon && abs(m12 - 0) < kEpsilon && abs(m13 - 0) < kEpsilon &&
               abs(m21 - 0) < kEpsilon && abs(m22 - 1) < kEpsilon && abs(m23 - 0) < kEpsilon &&
               abs(m31 - 0) < kEpsilon && abs(m32 - 0) < kEpsilon && abs(m33 - 1) < kEpsilon;
    }

    __device__ __host__ Vec3 Matrix3x3::get_normal() const
    {
        return transform(Vec3(1, 0, 0));
    }

    __device__ __host__ void Matrix3x3::print() const
    {
        printf(" Matrix3x3:\n");
        printf(" %f %f %f\n", m11, m12, m13);
        printf(" %f %f %f\n", m21, m22, m23);
        printf(" %f %f %f\n", m31, m32, m33);
    }

    __device__ __host__ Matrix3x3 Matrix3x3::make_identity()
    {
        return Matrix3x3(precision_t(1), precision_t(0), precision_t(0),
                         precision_t(0), precision_t(1), precision_t(0),
                         precision_t(0), precision_t(0), precision_t(1));
    }

    __device__ __host__ Matrix3x3 Matrix3x3::make_scale(const precision_t &scale1,
                                                        const precision_t &scale2,
                                                        const precision_t &scale3)
    {

        return Matrix3x3(scale1, precision_t(0), precision_t(0),
                         precision_t(0), scale2, precision_t(0),
                         precision_t(0), precision_t(0), scale3);
    }

    __device__ __host__ Matrix3x3 Matrix3x3::make_rotation_z(const precision_t &theta)
    {
        return Matrix3x3(cos(theta), -sin(theta), precision_t(0),
                         sin(theta), cos(theta), precision_t(0),
                         precision_t(0), precision_t(0), precision_t(1));
    }

    // Matrix4x4 类的实现
    __device__ __host__ Matrix4x4::Matrix4x4(const precision_t &m11_, const precision_t &m12_, const precision_t &m13_, const precision_t &m14_,
                                             const precision_t &m21_, const precision_t &m22_, const precision_t &m23_, const precision_t &m24_,
                                             const precision_t &m31_, const precision_t &m32_, const precision_t &m33_, const precision_t &m34_,
                                             const precision_t &m41_, const precision_t &m42_, const precision_t &m43_, const precision_t &m44_)
        : m11(m11_), m12(m12_), m13(m13_), m14(m14_),
          m21(m21_), m22(m22_), m23(m23_), m24(m24_),
          m31(m31_), m32(m32_), m33(m33_), m34(m34_),
          m41(m41_), m42(m42_), m43(m43_), m44(m44_)
    {
    }

    __device__ __host__ Matrix4x4 Matrix4x4::compose(const Matrix4x4 &mat) const
    {
        return Matrix4x4(
            m11 * mat.m11 + m12 * mat.m21 + m13 * mat.m31 + m14 * mat.m41,
            m11 * mat.m12 + m12 * mat.m22 + m13 * mat.m32 + m14 * mat.m42,
            m11 * mat.m13 + m12 * mat.m23 + m13 * mat.m33 + m14 * mat.m43,
            m11 * mat.m14 + m12 * mat.m24 + m13 * mat.m34 + m14 * mat.m44,
            m21 * mat.m11 + m22 * mat.m21 + m23 * mat.m31 + m24 * mat.m41,
            m21 * mat.m12 + m22 * mat.m22 + m23 * mat.m32 + m24 * mat.m42,
            m21 * mat.m13 + m22 * mat.m23 + m23 * mat.m33 + m24 * mat.m43,
            m21 * mat.m14 + m22 * mat.m24 + m23 * mat.m34 + m24 * mat.m44,
            m31 * mat.m11 + m32 * mat.m21 + m33 * mat.m31 + m34 * mat.m41,
            m31 * mat.m12 + m32 * mat.m22 + m33 * mat.m32 + m34 * mat.m42,
            m31 * mat.m13 + m32 * mat.m23 + m33 * mat.m33 + m34 * mat.m43,
            m31 * mat.m14 + m32 * mat.m24 + m33 * mat.m34 + m34 * mat.m44,
            m41 * mat.m11 + m42 * mat.m21 + m43 * mat.m31 + m44 * mat.m41,
            m41 * mat.m12 + m42 * mat.m22 + m43 * mat.m32 + m44 * mat.m42,
            m41 * mat.m13 + m42 * mat.m23 + m43 * mat.m33 + m44 * mat.m43,
            m41 * mat.m14 + m42 * mat.m24 + m43 * mat.m34 + m44 * mat.m44);
    }

    __device__ __host__ Matrix4x4 Matrix4x4::from_Matrix3x3(const Mat3x3 &mat)
    {
        return Matrix4x4(mat.m11, mat.m12, mat.m13, precision_t(0),
                         mat.m21, mat.m22, mat.m23, precision_t(0),
                         mat.m31, mat.m32, mat.m33, precision_t(0),
                         precision_t(0), precision_t(0), precision_t(0), precision_t(1));
    }

    __device__ __host__ Matrix4x4::Mat3x3 Matrix4x4::to_Matrix3x3() const
    {
        return Matrix3x3(m11, m12, m13,
                         m21, m22, m23,
                         m31, m32, m33);
    }

    __device__ __host__ Vec3 Matrix4x4::transform(const Vec3 &vec) const
    {
        return Vec3(m11 * vec.x + m12 * vec.y + m13 * vec.z + m14,
                  m21 * vec.x + m22 * vec.y + m23 * vec.z + m24,
                  m31 * vec.x + m32 * vec.y + m33 * vec.z + m34);
    }

    __device__ __host__ Vec3 Matrix4x4::get_normal() const
    {
        return to_Matrix3x3().transform(Vec3(-1, 0, 0)).norm();
    }

    __device__ __host__ bool Matrix4x4::is_identity() const
    {
      return abs(m11 - 1) < kEpsilon && abs(m12 - 0) < kEpsilon && abs(m13 - 0) < kEpsilon &&
             abs(m21 - 0) < kEpsilon && abs(m22 - 1) < kEpsilon && abs(m23 - 0) < kEpsilon &&
             abs(m31 - 0) < kEpsilon && abs(m32 - 0) < kEpsilon && abs(m33 - 1) < kEpsilon;
    }

    __device__ __host__ void Matrix4x4::print() const
    {
        printf("Matrix4x4:\n");
        printf(" %f %f %f %f\n", m11, m12, m13, m14);
        printf(" %f %f %f %f\n", m21, m22, m23, m24);
        printf(" %f %f %f %f\n", m31, m32, m33, m34);
        printf(" %f %f %f %f\n", m41, m42, m43, m44);
    }

    __device__ __host__ Matrix4x4 Matrix4x4::make_identity()
    {
        return Matrix4x4(precision_t(1), precision_t(0), precision_t(0), precision_t(0),
                         precision_t(0), precision_t(1), precision_t(0), precision_t(0),
                         precision_t(0), precision_t(0), precision_t(1), precision_t(0),
                         precision_t(0), precision_t(0), precision_t(0), precision_t(1));
    }

    __device__ __host__ Matrix4x4 Matrix4x4::make_scale(const Vec3 &vec)
    {
        return Matrix4x4(vec.x, precision_t(0), precision_t(0), precision_t(0),
                         precision_t(0), vec.y, precision_t(0), precision_t(0),
                         precision_t(0), precision_t(0), vec.z, precision_t(0),
                         precision_t(0), precision_t(0), precision_t(0), precision_t(1));
    }

    __device__ __host__ Matrix4x4 Matrix4x4::make_scale(const precision_t &scale1, const precision_t &scale2, const precision_t &scale3)
    {
        return Matrix4x4(scale1, precision_t(0), precision_t(0), precision_t(0),
                         precision_t(0), scale2, precision_t(0), precision_t(0),
                         precision_t(0), precision_t(0), scale3, precision_t(0),
                         precision_t(0), precision_t(0), precision_t(0), precision_t(1));
    }
    __device__ __host__ Matrix4x4 Matrix4x4::make_translation(const Vec3 &vec)
    {
        // printf("make_translation: %f, %f, %f\n", vec.x, vec.y, vec.z);
        return Matrix4x4(precision_t(1), precision_t(0), precision_t(0), vec.x,
                         precision_t(0), precision_t(1), precision_t(0), vec.y,
                         precision_t(0), precision_t(0), precision_t(1), vec.z,
                         precision_t(0), precision_t(0), precision_t(0), precision_t(1));
    }
    __device__ __host__ Matrix4x4 Matrix4x4::make_translation(const precision_t &x, const precision_t &y, const precision_t &z)
    {
        // printf("make_translation: %f, %f, %f\n", x, y, z);
        return Matrix4x4(precision_t(1), precision_t(0), precision_t(0), x,
                         precision_t(0), precision_t(1), precision_t(0), y,
                         precision_t(0), precision_t(0), precision_t(1), z,
                         precision_t(0), precision_t(0), precision_t(0), precision_t(1));
    }

    __device__ __host__ Matrix4x4 Matrix4x4::make_rotation_z(const precision_t &theta)
    {
        return Matrix4x4(cos(theta), -sin(theta), precision_t(0), precision_t(0),
                         sin(theta), cos(theta), precision_t(0), precision_t(0),
                         precision_t(0), precision_t(0), precision_t(1), precision_t(0),
                         precision_t(0), precision_t(0), precision_t(0), precision_t(1));
    }

    __device__ __host__ Vec3 lookup_table_functor::operator()(const size_t &index) const
    {
        const size_t ind_z = index / (size.x * size.y);
        const size_t ind_y = (index - ind_z * size.x * size.y) / size.x;
        const size_t ind_x = index - ind_z * size.x * size.y - ind_y * size.x;
        Vec3 ind_T(ind_x, ind_y, ind_z);
        auto transformed_spacing = direction.transform(spacing);
        return origin + ind_T * transformed_spacing;
    }

     __host__ Point3DVector Coordinate3D::make_lookup_table() const
    {

        size_t num_voxels = size.product();
        thrust::counting_iterator<size_t> index_begin(0);
        Point3DVector lookup_table(num_voxels);
        thrust::transform(thrust::device, index_begin, index_begin + num_voxels,
                          lookup_table.begin(), lookup_table_functor(origin, size, spacing, direction));
        return lookup_table;
    }

    __device__ __host__ void Coordinate3D::print() const
    {
        printf("origin:\n");
        origin.print();
        printf("size:\n");
        size.print();
        printf("spacing:\n");
        spacing.print();
        printf("direction:\n");
        direction.print();
    }

    template struct Value3<precision_t>;
    template struct Value3<int>;
    template struct Value3<size_t>;

} // namespace common

NMRECON_NAMESPACE_END
