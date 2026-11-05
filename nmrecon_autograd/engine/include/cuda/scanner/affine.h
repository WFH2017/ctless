
#ifndef NMRECON_SCANNER_AFFINE_H_
#define NMRECON_SCANNER_AFFINE_H_

#include "common/config.h"
#include "reconstruction/value3.h"

NMRECON_NAMESPACE_BEGIN
namespace scanner {
typedef nmrecon::reconstruction::Value3<float> Vector3D;
const float eps_ =1e-6;

struct Affine3D {
    Affine3D() = default;
    __device__ Affine3D(float m11, float m12, float m13, float m14,
                                 float m21, float m22, float m23, float m24,
                                 float m31, float m32, float m33, float m34,
                                 float m41, float m42, float m43, float m44)
        : m11(m11), m12(m12), m13(m13), m14(m14), m21(m21), m22(m22), m23(m23),
          m24(m24), m31(m31), m32(m32), m33(m33), m34(m34), m41(m41), m42(m42),
          m43(m43), m44(m44) {
    }

    Affine3D(const h_vec_f32 &data_)
        : Affine3D(data_[0], data_[1], data_[2], data_[3], data_[4], data_[5],
                   data_[6], data_[7], data_[8], data_[9], data_[10], data_[11],
                   data_[12], data_[13], data_[14], data_[15]) {
        assert(data_.size() == 16);
    }

    Affine3D(const float *data_)
        : Affine3D(data_[0], data_[1], data_[2], data_[3], data_[4], data_[5],
                   data_[6], data_[7], data_[8], data_[9], data_[10], data_[11],
                   data_[12], data_[13], data_[14], data_[15]) {}

    Affine3D(const std::vector<float> &data_) : Affine3D(h_vec_f32(data_)) {}

    __device__ Affine3D apply(const Affine3D &aff) const {
        return {m11 * aff.m11 + m12 * aff.m21 + m13 * aff.m31 + m14 * aff.m41,
                m11 * aff.m12 + m12 * aff.m22 + m13 * aff.m32 + m14 * aff.m42,
                m11 * aff.m13 + m12 * aff.m23 + m13 * aff.m33 + m14 * aff.m43,
                m11 * aff.m14 + m12 * aff.m24 + m13 * aff.m34 + m14 * aff.m44,
                m21 * aff.m11 + m22 * aff.m21 + m23 * aff.m31 + m24 * aff.m41,
                m21 * aff.m12 + m22 * aff.m22 + m23 * aff.m32 + m24 * aff.m42,
                m21 * aff.m13 + m22 * aff.m23 + m23 * aff.m33 + m24 * aff.m43,
                m21 * aff.m14 + m22 * aff.m24 + m23 * aff.m34 + m24 * aff.m44,
                m31 * aff.m11 + m32 * aff.m21 + m33 * aff.m31 + m34 * aff.m41,
                m31 * aff.m12 + m32 * aff.m22 + m33 * aff.m32 + m34 * aff.m42,
                m31 * aff.m13 + m32 * aff.m23 + m33 * aff.m33 + m34 * aff.m43,
                m31 * aff.m14 + m32 * aff.m24 + m33 * aff.m34 + m34 * aff.m44,
                m41 * aff.m11 + m42 * aff.m21 + m43 * aff.m31 + m44 * aff.m41,
                m41 * aff.m12 + m42 * aff.m22 + m43 * aff.m32 + m44 * aff.m42,
                m41 * aff.m13 + m42 * aff.m23 + m43 * aff.m33 + m44 * aff.m43,
                m41 * aff.m14 + m42 * aff.m24 + m43 * aff.m34 + m44 * aff.m44};
    }

    __device__ Vector3D apply(const Vector3D &vec) {
        // if (_isScale) {
        //   return {m11 * vec.x, m22 * vec.y, m33 * vec.z};
        // }
        return {m11 * vec.x + m12 * vec.y + m13 * vec.z + m14,
                m21 * vec.x + m22 * vec.y + m23 * vec.z + m24,
                m31 * vec.x + m32 * vec.y + m33 * vec.z + m34};
    }

    __device__ Affine3D operator+(const Affine3D &aff) {
        return {m11 + aff.m11, m12 + aff.m12, m13 + aff.m13, m14 + aff.m14,
                m21 + aff.m31, m22 + aff.m32, m23 + aff.m23, m24 + aff.m24,
                m31 + aff.m31, m32 + aff.m32, m33 + aff.m33, m34 + aff.m34,
                m41 + aff.m41, m42 + aff.m42, m43 + aff.m43, m44 + aff.m44};
    }
    __device__ Affine3D operator*(const float scale) {
        return {m11 * scale, m12 * scale, m13 * scale, m14 * scale,
                m21 * scale, m22 * scale, m23 * scale, m24 * scale,
                m31 * scale, m32 * scale, m33 * scale, m34 * scale,
                m41 * scale, m42 * scale, m43 * scale, m44 * scale};
    }
    __device__ Vector3D GetCenter() const { return {m14, m24, m34}; }
    __device__ Affine3D operator-(const Vector3D &vec) {
        return {m11, m12, m13, m14 - vec.x, m21, m22, m23, m24 - vec.y,
                m31, m32, m33, m34 - vec.z, m41, m42, m43, m44};
    }
    static __device__ Affine3D MakeScaling(const float sx,
                                                    const float sy,
                                                    const float sz) {
        return {sx,   0.0f, 0.0f, 0.0f, 0.0f, sy,   0.0f, 0.0f,
                0.0f, 0.0f, sz,   0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    }

    static __device__ Affine3D Makefloatranslation(const float tx,
                                                            const float ty,
                                                            const float tz) {
        return {1.0f, 0.0f, 0.0f, tx, 0.0f, 1.0f, 0.0f, ty,
                0.0f, 0.0f, 1.0f, tz, 0.0f, 0.0f, 0.0f, 1.0f};
    }

    static __device__ Affine3D MakeRotationZ(const float theta) {
        return {cosf(theta), -sinf(theta), 0.0f, 0.0f, sinf(theta), cosf(theta),
                0.0f,        0.0f,         0.0f, 0.0f, 1.0f,        0.0f,
                0.0f,        0.0f,         0.0f, 1.0f};
    }

    std::vector<float> ToStl() const {
        return {m11, m12, m13, m14, m21, m22, m23, m24,
                m31, m32, m33, m34, m41, m42, m43, m44};
    }
    // h_vec_f32 floatoVec() const { return this->floatoStl(); }
    static int GetNum() { return 16; }

    friend std::ostream &operator<<(std::ostream &os, Affine3D &affine) {

        os << affine.m11 << " " << affine.m12 << " " << affine.m13 << " "
           << affine.m14 << std::endl;
        os << affine.m21 << " " << affine.m22 << " " << affine.m23 << " "
           << affine.m24 << std::endl;
        os << affine.m31 << " " << affine.m32 << " " << affine.m33 << " "
           << affine.m34 << std::endl;
        os << affine.m41 << " " << affine.m42 << " " << affine.m43 << " "
           << affine.m44 << std::endl;
        return os;
    }

    __inline__ __device__ Affine3D GetInvert() const {
        // if (_isScale) {
        //   return Affine3D::MakeScaling(1.0f / m11, 1.0f / m22, 1.0f / m33);
        // }

        float t11 = m23 * m34 * m42 - m24 * m33 * m42 + m24 * m32 * m43 -
                    m22 * m34 * m43 - m23 * m32 * m44 + m22 * m33 * m44,
              t12 = m14 * m33 * m42 - m13 * m34 * m42 - m14 * m32 * m43 +
                    m12 * m34 * m43 + m13 * m32 * m44 - m12 * m33 * m44,
              t13 = m13 * m24 * m42 - m14 * m23 * m42 + m14 * m22 * m43 -
                    m12 * m24 * m43 - m13 * m22 * m44 + m12 * m23 * m44,
              t14 = m14 * m23 * m32 - m13 * m24 * m32 - m14 * m22 * m33 +
                    m12 * m24 * m33 + m13 * m22 * m34 - m12 * m23 * m34;

        float det = m11 * t11 + m21 * t12 + m31 * t13 + m41 * t14;

        if (det < eps_ && det > -eps_)
            return Affine3D();

        float detInv = 1 / det;

        float new_m11 = t11 * detInv;
        float new_m21 = (m24 * m33 * m41 - m23 * m34 * m41 - m24 * m31 * m43 +
                         m21 * m34 * m43 + m23 * m31 * m44 - m21 * m33 * m44) *
                        detInv;
        float new_m31 = (m22 * m34 * m41 - m24 * m32 * m41 + m24 * m31 * m42 -
                         m21 * m34 * m42 - m22 * m31 * m44 + m21 * m32 * m44) *
                        detInv;
        float new_m41 = (m23 * m32 * m41 - m22 * m33 * m41 - m23 * m31 * m42 +
                         m21 * m33 * m42 + m22 * m31 * m43 - m21 * m32 * m43) *
                        detInv;

        float new_m12 = t12 * detInv;
        float new_m22 = (m13 * m34 * m41 - m14 * m33 * m41 + m14 * m31 * m43 -
                         m11 * m34 * m43 - m13 * m31 * m44 + m11 * m33 * m44) *
                        detInv;
        float new_m32 = (m14 * m32 * m41 - m12 * m34 * m41 - m14 * m31 * m42 +
                         m11 * m34 * m42 + m12 * m31 * m44 - m11 * m32 * m44) *
                        detInv;
        float new_m42 = (m12 * m33 * m41 - m13 * m32 * m41 + m13 * m31 * m42 -
                         m11 * m33 * m42 - m12 * m31 * m43 + m11 * m32 * m43) *
                        detInv;

        float new_m13 = t13 * detInv;
        float new_m23 = (m14 * m23 * m41 - m13 * m24 * m41 - m14 * m21 * m43 +
                         m11 * m24 * m43 + m13 * m21 * m44 - m11 * m23 * m44) *
                        detInv;
        float new_m33 = (m12 * m24 * m41 - m14 * m22 * m41 + m14 * m21 * m42 -
                         m11 * m24 * m42 - m12 * m21 * m44 + m11 * m22 * m44) *
                        detInv;
        float new_m43 = (m13 * m22 * m41 - m12 * m23 * m41 - m13 * m21 * m42 +
                         m11 * m23 * m42 + m12 * m21 * m43 - m11 * m22 * m43) *
                        detInv;

        float new_m14 = t14 * detInv;
        float new_m24 = (m13 * m24 * m31 - m14 * m23 * m31 + m14 * m21 * m33 -
                         m11 * m24 * m33 - m13 * m21 * m34 + m11 * m23 * m34) *
                        detInv;
        float new_m34 = (m14 * m22 * m31 - m12 * m24 * m31 - m14 * m21 * m32 +
                         m11 * m24 * m32 + m12 * m21 * m34 - m11 * m22 * m34) *
                        detInv;
        float new_m44 = (m12 * m23 * m31 - m13 * m22 * m31 + m13 * m21 * m32 -
                         m11 * m23 * m32 - m12 * m21 * m33 + m11 * m22 * m33) *
                        detInv;
        return {new_m11, new_m12, new_m13, new_m14, new_m21, new_m22,
                new_m23, new_m24, new_m31, new_m32, new_m33, new_m34,
                new_m41, new_m42, new_m43, new_m44};
    }
    float m11 = 1.0f, m12 = 0.0f, m13 = 0.0f, m14 = 0.0f, m21 = 0.0f,
          m22 = 1.0f, m23 = 0.0f, m24 = 0.0f, m31 = 0.0f, m32 = 0.0f,
          m33 = 1.0f, m34 = 0.0f, m41 = 0.0f, m42 = 0.0f, m43 = 0.0f,
          m44 = 1.0f;
    // static Affine3D LoadJson(json j);
};


typedef thrust::device_vector<Affine3D> d_vec_aff;
typedef thrust::host_vector<Affine3D> h_vec_aff;

// typedef thrust::host_vector<Affine3D> Affine3DList; 
}
NMRECON_NAMESPACE_END
#endif // !NMRECON_SCANNER_AFFINE_H_