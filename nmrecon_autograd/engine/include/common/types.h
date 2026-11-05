#pragma once

/*!
  \file types.h
  \ingroup Config
  \brief

  \author Gaoyu Chen: chengaoyu2013@gmail.com
  \author
*/

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "common/namespace.h"

NMRECON_NAMESPACE_BEGIN

using thrust::device_vector;
using thrust::host_vector;

// basic types on device
typedef device_vector<bool> d_vec_bool;
typedef device_vector<float> d_vec_f32;
typedef device_vector<int> d_vec_i32;
typedef device_vector<unsigned short> d_vec_u16;
typedef device_vector<unsigned int> d_vec_u32;
// typedef device_vector<IndexType> d_vec_idx;

// basic types on host
typedef host_vector<bool> h_vec_bool;
typedef host_vector<float> h_vec_f32;
typedef host_vector<int> h_vec_i64;
typedef host_vector<unsigned short> h_vec_u16;
typedef host_vector<unsigned int> h_vec_u32;
// typedef host_vector<IndexType> h_vec_idx;

typedef unsigned short ushort;

// precision type, default is float, if you want to use double, please change it to double
typedef float precision_t;

NMRECON_NAMESPACE_END
