/*!
  \file
  \ingroup
  \brief

  \author Gaoyu Chen: chengaoyu2013@gmail.com
  \author
*/
#pragma once
#ifndef NMRECON_FUNCTOR_H
#define NMRECON_FUNCTOR_H

#include <cmath>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include "common/config.h"

using namespace thrust::placeholders;

NMRECON_NAMESPACE_BEGIN
namespace common
{
    namespace functor
    {

        template <typename T>
        struct generator
        {
            T a;
            generator(T a_) : a(a_) {}
            __host__ __device__ T operator()() { return a; }
        };

        template <typename T>
        struct gt_functor : public thrust::unary_function<T, T>
        {
            T a;
            gt_functor(T a_) : a(a_) {}
            __host__ __device__ T operator()(T x)
            {
                if (x > a)
                {
                    return 1.f;
                }
                else
                {
                    return 0.f;
                }
            }
        };

        template <typename T>
        struct gt2_functor : public thrust::binary_function<T, T, T>
        {
            __host__ __device__ T operator()(T x, T y)
            {
                if (x > y)
                {
                    return 1.f;
                }
                else
                {
                    return 0.f;
                }
            }
        };

        template <typename T>
        struct lt_functor : public thrust::unary_function<T, T>
        {
            T a;
            lt_functor(T a_) : a(a_) {}
            __host__ __device__ T operator()(T x)
            {
                if (x < a)
                {
                    return 1.f;
                }
                else
                {
                    return 0.f;
                }
            }
        };

        template <typename T>
        struct lt2_functor : public thrust::binary_function<T, T, T>
        {
            __host__ __device__ T operator()(T x, T y)
            {
                if (x < y)
                {
                    return 1.f;
                }
                else
                {
                    return 0.f;
                }
            }
        };

        template <typename T>
        struct ge_functor : public thrust::unary_function<T, T>
        {
            T a;
            ge_functor(T a_) : a(a_) {}
            __host__ __device__ T operator()(T x)
            {
                if (x >= a)
                {
                    return 1.f;
                }
                else
                {
                    return 0.f;
                }
            }
        };

        template <typename T>
        struct ge2_functor : public thrust::binary_function<T, T, T>
        {
            __host__ __device__ T operator()(T x, T y)
            {
                if (x >= y)
                {
                    return 1.f;
                }
                else
                {
                    return 0.f;
                }
            }
        };

        template <typename T>
        struct le_functor : public thrust::unary_function<T, T>
        {
            T a;
            le_functor(T a_) : a(a_) {}
            __host__ __device__ T operator()(T x)
            {
                if (x <= a)
                {
                    return 1.f;
                }
                else
                {
                    return 0.f;
                }
            }
        };

        template <typename T>
        struct le2_functor : public thrust::binary_function<T, T, T>
        {
            __host__ __device__ T operator()(T x, T y)
            {
                if (x <= y)
                {
                    return 1.f;
                }
                else
                {
                    return 0.f;
                }
            }
        };

        template <typename T>
        struct eq_functor : public thrust::unary_function<T, T>
        {
            T a;
            eq_functor(T a_) : a(a_) {}
            __host__ __device__ T operator()(T x)
            {
                if (x == a)
                {
                    return 1.f;
                }
                else
                {
                    return 0.f;
                }
            }
        };

        template <typename T>
        struct eq2_functor : public thrust::binary_function<T, T, T>
        {
            __host__ __device__ T operator()(T x, T y)
            {
                if (x == y)
                {
                    return 1.f;
                }
                else
                {
                    return 0.f;
                }
            }
        };

        template <typename T>
        struct abs_functor : public thrust::unary_function<T, T>
        {
            __host__ __device__ T operator()(const T &z)
            {
                if (z >= 0.f)
                {
                    return z;
                }
                else
                {
                    return -z;
                }
            }
        };

        template <typename T>
        struct pow_functor : public thrust::binary_function<T, T, T>
        {
            T val;
            pow_functor(const T val_) : val(val_) {}

            __host__ __device__ T operator()(const T &z) { return std::pow(z, val); }
        };

        template <typename T>
        struct exp_functor : public thrust::unary_function<T, T>
        {
            __host__ __device__ T operator()(const T &z) { return std::exp(z); }
        };

        template <typename T>
        struct exp2_functor : public thrust::binary_function<T, T, T>
        {
            T base;
            exp2_functor(const T base_ = 2) : base(base_) {}

            __host__ __device__ T operator()(const T &z) { return std::pow(base, z); }
        };

        template <typename T>
        struct log_functor : public thrust::binary_function<T, T, T>
        {
            T log_base;
            log_functor(const T base_ = 1) : log_base(std::log(base_)) {}

            __host__ __device__ T operator()(const T &z)
            {
                if (log_base == 0)
                    return std::log(z);
                else
                    return std::log(z) / log_base;
            }
        };

        template <typename T>
        struct true_divide_functor : public thrust::binary_function<T, T, T>
        {
            T divider;
            T _default;
            true_divide_functor(const T divider_ = 1, const T default_ = 0)
                : divider(divider_), _default(default_)
            {
                if (divider_ == 0)
                {
                    fprintf(stderr, "Divider is set to 0 in file '%s' in line %i.\n",
                            __FILE__, __LINE__);
                }
            }

            __host__ __device__ T operator()(const T &z) { return z / divider; }

            __host__ __device__ T operator()(const T &z1, const T &z2)
            {
                if (z2 < 1e-6)
                {
                    return 0.0f;
                }
                else
                {
                    return z1 / z2;
                }
                // return z1/(z2+(T)1e-6);
            }
        };

        template <typename T>
        struct divide_thres_functor : public thrust::binary_function<T, T, T>
        {
            T threshold;
            T _default;
            divide_thres_functor(const T threshold_ = 0, const T default_ = 0)
                : threshold(threshold_), _default(default_) {}

            __host__ __device__ T operator()(const T &z1, const T &z2)
            {
                if (z2 < threshold)
                {
                    return T(_default);
                }
                else
                {
                    return z1 / z2;
                }
            }
        };

        template <typename T>
        struct remove_nan_functor : public thrust::unary_function<T, T>
        {
            __host__ __device__ T operator()(const T &x)
            {
                if (isnan(x))
                {
                    return 0.f;
                }
                else
                {
                    return x;
                }
            }
        };

        template <typename T>
        struct remove_inf_functor : public thrust::unary_function<T, T>
        {
            __host__ __device__ T operator()(const T &x)
            {
                if (isinf(x))
                {
                    return 0.f;
                }
                else
                {
                    return x;
                }
            }
        };
        template <typename T>
        struct remove_negative_functor : public thrust::unary_function<T, T>
        {
            __host__ __device__ T operator()(const T &x)
            {
                if (x < 0)
                {
                    return T(0);
                }
                else
                {
                    return x;
                }
            }
        };

        template <typename T>
        struct search_lookup_functor : public thrust::binary_function<T, T, T>
        {
            thrust::device_vector<T> lookup_table;
            search_lookup_functor(const thrust::device_vector<T> &lookup_table_)
                : lookup_table(lookup_table_) {}

            __host__ __device__ T operator()(const int &ind)
            {
                if (ind < 0 || ind >= lookup_table.size())
                {
                    return T(0);
                }
                else
                {
                    return lookup_table[ind];
                }
            }
        };

    } // end namespace functor
} // end namespace common
NMRECON_NAMESPACE_END
#endif // INCLUDE_FUNCTOR_H
