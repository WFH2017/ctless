#pragma once

#include <thrust/host_vector.h>
#include "common/namespace.h"

NMRECON_NAMESPACE_BEGIN
namespace common
{
    namespace host_arithmetic
    {

        template <typename T>
        thrust::host_vector<T> operator+(const thrust::host_vector<T> &vec, const thrust::host_vector<T> &vec2);

        template <typename T, typename T2>
        thrust::host_vector<T> operator+(const thrust::host_vector<T> &vec, const T2 &val);

        template <typename T, typename T2>
        thrust::host_vector<T> operator+(const T2 &val, const thrust::host_vector<T> &vec);

        template <typename T>
        thrust::host_vector<T> operator-(const thrust::host_vector<T> &vec);

        template <typename T>
        thrust::host_vector<T> operator-(const thrust::host_vector<T> &vec, const thrust::host_vector<T> &vec2);

        template <typename T, typename T2>
        thrust::host_vector<T> operator-(const thrust::host_vector<T> &vec, const T2 &val);

        template <typename T, typename T2>
        thrust::host_vector<T> operator-(const T2 &val, const thrust::host_vector<T> &vec);

        template <typename T>
        thrust::host_vector<T> operator*(const thrust::host_vector<T> &vec, const thrust::host_vector<T> &vec2);

        template <typename T, typename T2>
        thrust::host_vector<T> operator*(const thrust::host_vector<T> &vec, const T2 &val);

        template <typename T>
        thrust::host_vector<T> operator/(const thrust::host_vector<T> &vec, const thrust::host_vector<T> &vec2);

        template <typename T, typename T2>
        thrust::host_vector<T> operator/(const thrust::host_vector<T> &vec, const T2 &val);

        template <typename T>
        thrust::host_vector<T> true_div(const thrust::host_vector<T> &vec, const thrust::host_vector<T> &val);

        template <typename T>
        thrust::host_vector<T> div_with_threshold(const thrust::host_vector<T> &vec, const thrust::host_vector<T> &val, const T threshold, const T _default);

        template <typename T>
        thrust::host_vector<T> absolute(const thrust::host_vector<T> &vec);

        template <typename T>
        thrust::host_vector<T> pow(const thrust::host_vector<T> &vec, const float val);

        template <typename T>
        thrust::host_vector<T> sqrt(const thrust::host_vector<T> &vec);

        template <typename T>
        thrust::host_vector<T> exp(const thrust::host_vector<T> &vec);

        template <typename T>
        thrust::host_vector<T> log(const thrust::host_vector<T> &vec, const T base);

        template <typename T>
        thrust::host_vector<T> remove_nan(const thrust::host_vector<T> &vec);

        template <typename T>
        thrust::host_vector<T> remove_inf(const thrust::host_vector<T> &vec);

        template <typename T>
        thrust::host_vector<T> remove_negative(const thrust::host_vector<T> &vec);

        template <typename TInd, typename T>
        thrust::host_vector<T> search_lookup(const thrust::host_vector<TInd> &index, const thrust::host_vector<T> &lookup_table);

        template <typename T1, typename T2>
        thrust::host_vector<T2> convert(const thrust::host_vector<T1> &data);

    } // end namespace host_arithmetic
} // end namespace common
NMRECON_NAMESPACE_END
