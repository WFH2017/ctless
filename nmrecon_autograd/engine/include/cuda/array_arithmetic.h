//
// Created by Minghao Guo on 2021/3/1.
//
#pragma once
#ifndef NMRECON_DEVICE_ARITHMETIC_H_
#define NMRECON_DEVICE_ARITHMETIC_H_

#include <iterator>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "functor.h"

#include "common/config.h"
#include "array.h"

// using namespace thrust;
using namespace thrust::placeholders;

NMRECON_NAMESPACE_BEGIN
namespace common{

template <typename T, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T,Size, Cont> operator+( const Array<T,Size, Cont> &vec,
                                const Array<T,Size, Cont> &vec2 ) {
    assert( vec.size() == vec2.size() );
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform(
        vec.begin(), vec.end(), vec2.begin(), vec_out.begin(), _1 + _2 );
    return vec_out;
}

template <typename T, typename T2, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont>  operator+( const Array<T,Size, Cont>  &vec, const T2 &val ) {
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform( vec.begin(), vec.end(), vec_out.begin(), _1 + val );
    return vec_out;
}

template <typename T, typename T2, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> operator+( const T2 &val, const Array<T, Size, Cont> &vec ) {
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform( vec.begin(), vec.end(), vec_out.begin(), _1 + val );
    return vec_out;
}

template <typename T, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> operator-( const Array<T, Size, Cont> &vec ) {
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform( vec.begin(), vec.end(), vec_out.begin(), -_1 );
    return vec_out;
}

template <typename T, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> operator-( const Array<T, Size, Cont> &vec,
                                const Array<T, Size, Cont> &vec2 ) {
    assert( vec.size() == vec2.size() );
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform(
        vec.begin(), vec.end(), vec2.begin(), vec_out.begin(), _1 - _2 );
    return vec_out;
}

template <typename T, typename T2, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> operator-( const Array<T, Size, Cont> &vec, const T2 &val ) {
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform( vec.begin(), vec.end(), vec_out.begin(), _1 - val );
    return vec_out;
}

template <typename T, typename T2, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> operator-( const T2 &val, const Array<T, Size, Cont> &vec ) {
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform( vec.begin(), vec.end(), vec_out.begin(), val - _1 );
    return vec_out;
}

template <typename T, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> operator*( const Array<T, Size, Cont> &vec,
                            const Array<T, Size, Cont> &vec2 ) {
    assert( vec.size() == vec2.size() );
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform(
        vec.begin(), vec.end(), vec2.begin(), vec_out.begin(), _1 * _2 );
    return vec_out;
}

template <typename T, typename T2, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> operator*( const Array<T, Size, Cont> &vec, const T2 &val ) {
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform( vec.begin(), vec.end(), vec_out.begin(), _1 * val );
    return vec_out;
}

template <typename T, typename T2, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> operator*( const T2 &val, const Array<T, Size, Cont> &vec ) {
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform( vec.begin(), vec.end(), vec_out.begin(), val * _1 );
    return vec_out;
}

template <typename T, typename T2, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> operator*( const Array<T, Size, Cont> &vec, const Array<T2, Size, Cont> &vec2 ) {
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform( vec.begin(), vec.end(), vec_out.begin(), _1 * _2 );
    return vec_out;
}


template <typename T, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> operator/( const Array<T, Size, Cont> &vec,
                            const Array<T, Size, Cont> &vec2 ) {
    assert( vec.size() == vec2.size() );
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform(
        vec.begin(), vec.end(), vec2.begin(), vec_out.begin(), _1 / _2 );
    return vec_out;
}

template <typename T, typename T2, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> operator/( const Array<T, Size, Cont> &vec, const T2 &val ) {
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform( vec.begin(), vec.end(), vec_out.begin(), _1 / val );
    return vec_out;
}

template <typename T, typename T2, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> operator/( const T2 &val, const Array<T, Size, Cont> &vec ) {
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform( vec.begin(), vec.end(), vec_out.begin(), val / _1 );
    return vec_out;
}


template <typename T, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> operator%( const Array<T, Size, Cont> &vec, const T &val ) {
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform( vec.begin(), vec.end(), vec_out.begin(), _1 % val );
    return vec_out;
}

template <typename T, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> operator%( const T &val, const Array<T, Size, Cont> &vec ) {
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform( vec.begin(), vec.end(), vec_out.begin(), val % _1 );
    return vec_out;
}

template <typename T, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> true_div( const Array<T, Size, Cont> &vec,
                           const Array<T, Size, Cont> &val ) {
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform( vec.begin(),
                       vec.end(),
                       val.begin(),
                       vec_out.begin(),
                       functor::true_divide_functor<T>() );
    return vec_out;
}

template <typename T, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> div_with_threshold( const Array<T, Size, Cont> &vec,
                                     const Array<T, Size, Cont> &val,
                                     const T threshold,
                                     const T _default ) {
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform( vec.begin(),
                       vec.end(),
                       val.begin(),
                       vec_out.begin(),
                       functor::divide_thres_functor<T>( threshold, _default ) );
    return vec_out;
}

template <typename T, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector> 
Array<T, Size, Cont> absolute( const Array<T, Size, Cont> &vec ) {
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform(
        vec.begin(), vec.end(), vec_out.begin(), functor::abs_functor<T>() );
    return vec_out;
}

template <typename T, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> pow( const Array<T, Size, Cont> &vec, const float val ) {
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform(
        vec.begin(), vec.end(), vec_out.begin(), functor::pow_functor<T>( val ) );
    return vec_out;
}

template <typename T, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector> 
Array<T, Size, Cont> sqrt( const Array<T, Size, Cont> &vec ) {
    return pow<T>( vec, 0.5 );
}

template <typename T, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> exp( const Array<T, Size, Cont> &vec ) {
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform(
        vec.begin(), vec.end(), vec_out.begin(), functor::exp_functor<T>() );
    return vec_out;
}

template <typename T, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> exp2( const Array<T, Size, Cont> &vec, const T base ) {
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform(
        vec.begin(), vec.end(), vec_out.begin(), functor::exp2_functor<T>( base ) );
    return vec_out;
}

template <typename T, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> log( const Array<T, Size, Cont> &vec, const T base = 1 ) {
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform(
        vec.begin(), vec.end(), vec_out.begin(), functor::log_functor<T>( base ) );
    return vec_out;
}

template <typename T, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> remove_nan( const Array<T, Size, Cont> &vec ) {
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform(
        vec.begin(), vec.end(), vec_out.begin(), functor::remove_nan_functor<T>() );
    return vec_out;
}

template <typename T, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> remove_inf( const Array<T, Size, Cont> &vec ) {
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform(
        vec.begin(), vec.end(), vec_out.begin(), functor::remove_inf_functor<T>() );
    return vec_out;
}

template <typename T, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> remove_negative( const Array<T, Size, Cont> &vec ) {
    Array<T, Size, Cont> vec_out = Array<T, Size, Cont>( vec.size() );
    thrust::transform(
        vec.begin(), vec.end(), vec_out.begin(), functor::remove_negative_functor<T>() );
    return vec_out;
}

template <typename TInd, typename T, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T, Size, Cont> search_lookup( const Array<TInd, Size, Cont> &index,
                                const Array<T, Size, Cont> &lookup_table ) {
    auto iter = thrust::make_permutation_iterator( lookup_table.begin(),
                                                   index.begin() );
    return Array<T, Size, Cont>( iter, iter + index.size() );
}

template <typename T1, typename T2, std::size_t Size, 
template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
Array<T2, Size, Cont> convert( const Array<T1, Size, Cont> &data ) {
    return Array<T2, Size, Cont>( data.begin(), data.end() );
}



} // end namespace common
NMRECON_NAMESPACE_END

#endif // NMRECON_DEVICE_ARITHMETIC_H
