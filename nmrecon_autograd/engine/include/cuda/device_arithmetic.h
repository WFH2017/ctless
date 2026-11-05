#pragma once

#include <thrust/device_vector.h>
#include "common/namespace.h"

NMRECON_NAMESPACE_BEGIN
namespace common
{
    namespace device_arithmetic
    {

        template <typename T>
        thrust::device_vector<T> operator+(const thrust::device_vector<T> &vec, const thrust::device_vector<T> &vec2);

        template <typename T, typename T2>
        thrust::device_vector<T> operator+(const thrust::device_vector<T> &vec, const T2 &val);

        template <typename T, typename T2>
        thrust::device_vector<T> operator+(const T2 &val, const thrust::device_vector<T> &vec);

        template <typename T>
        thrust::device_vector<T> operator-(const thrust::device_vector<T> &vec);

        template <typename T>
        thrust::device_vector<T> operator-(const thrust::device_vector<T> &vec, const thrust::device_vector<T> &vec2);

        template <typename T, typename T2>
        thrust::device_vector<T> operator-(const thrust::device_vector<T> &vec, const T2 &val);

        template <typename T, typename T2>
        thrust::device_vector<T> operator-(const T2 &val, const thrust::device_vector<T> &vec);

        template <typename T>
        thrust::device_vector<T> operator*(const thrust::device_vector<T> &vec, const thrust::device_vector<T> &vec2);

        template <typename T, typename T2>
        thrust::device_vector<T> operator*(const thrust::device_vector<T> &vec, const T2 &val);

        template <typename T>
        thrust::device_vector<T> operator/(const thrust::device_vector<T> &vec, const thrust::device_vector<T> &vec2);

        template <typename T, typename T2>
        thrust::device_vector<T> operator/(const thrust::device_vector<T> &vec, const T2 &val);

        template <typename T>
        thrust::device_vector<T> true_div(const thrust::device_vector<T> &vec, const thrust::device_vector<T> &val);

        template <typename T>
        thrust::device_vector<T> div_with_threshold(const thrust::device_vector<T> &vec, const thrust::device_vector<T> &val, const T threshold, const T _default);

        template <typename T>
        thrust::device_vector<T> absolute(const thrust::device_vector<T> &vec);

        template <typename T>
        thrust::device_vector<T> pow(const thrust::device_vector<T> &vec, const float val);

        template <typename T>
        thrust::device_vector<T> sqrt(const thrust::device_vector<T> &vec);

        template <typename T>
        thrust::device_vector<T> exp(const thrust::device_vector<T> &vec);

        template <typename T>
        thrust::device_vector<T> log(const thrust::device_vector<T> &vec, const T base);

        template <typename T>
        thrust::device_vector<T> remove_nan(const thrust::device_vector<T> &vec);

        template <typename T>
        thrust::device_vector<T> remove_inf(const thrust::device_vector<T> &vec);

        template <typename T>
        thrust::device_vector<T> remove_negative(const thrust::device_vector<T> &vec);

        template <typename TInd, typename T>
        thrust::device_vector<T> search_lookup(const thrust::device_vector<TInd> &index, const thrust::device_vector<T> &lookup_table);

        template <typename T1, typename T2>
        thrust::device_vector<T2> convert(const thrust::device_vector<T1> &data);
        
        template <typename T>
        T find_min_positive(const thrust::device_vector<T> &vec);
        

    } // end namespace device_arithmetic
} // end namespace common
NMRECON_NAMESPACE_END
