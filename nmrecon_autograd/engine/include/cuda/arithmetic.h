
#pragma once
#ifndef NMRECON_ARITHMETIC_H_

#include <iterator>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "common/config.h"

using namespace thrust::placeholders;
using thrust::host_vector;

NMRECON_NAMESPACE_BEGIN
namespace common
{

namespace arithmetic
{

/*!
  \brief Add two vectors or a vector and a scalar.
  \param vec The first vector.
  \param vec2 The second vector.
  \return The sum of the two vectors.
*/
template <typename Vector = host_vector<float>>
Vector operator+(const Vector &vec, const Vector &vec2)
{
  assert(vec.size() == vec2.size());
  Vector vec_out = Vector(vec.size());
  thrust::transform(vec.begin(), vec.end(), vec2.begin(), vec_out.begin(), _1 + _2);
  return vec_out;
}

/*!
  \brief Add a vector and a scalar.
  \param vec The vector.
  \param val The scalar.
  \return The sum of the vector and the scalar.
*/
template <typename T, typename Vector= host_vector<float>>
Vector operator+(const Vector &vec, const T &val)
{
  Vector vec_out = Vector(vec.size());
  thrust::transform(vec.begin(), vec.end(), vec_out.begin(), _1 + val);
  return vec_out;
}

/*!
  \brief Add a vector and a scalar.
  \param vec The vector.
  \param val The scalar.
  \return The sum of the vector and the scalar.
*/
template < typename T=float, typename Vector= host_vector<float>>
Vector operator+(const T &val, const Vector &vec)
{
  Vector vec_out = Vector(vec.size());
  thrust::transform(vec.begin(), vec.end(), vec_out.begin(), _2 + val);
  return vec_out;
}


/*!
  \brief Negate a vector.
  \param vec The vector.
  \return The negated vector.
*/
template <typename Vector>
Vector operator-(const Vector &vec)
{
  Vector vec_out = Vector(vec.size());
  thrust::transform(vec.begin(), vec.end(), vec_out.begin(), -_1);
  return vec_out;
}

/*!
  \brief Subtract two vectors.
  \param vec The first vector.
  \param vec2 The second vector.
  \return The difference of the two vectors.
*/
template <typename Vector>
Vector operator-(const Vector &vec, const Vector &vec2)
{
  assert(vec.size() == vec2.size());
  Vector vec_out = Vector(vec.size());
  thrust::transform(vec.begin(), vec.end(), vec2.begin(), vec_out.begin(), _1 - _2);
  return vec_out;
}

/*!
  \brief Subtract a scalar from a vector.
  \param vec The vector.
  \param val The scalar.
  \return The difference of the vector and the scalar.
*/
template <typename T, typename Vector>
Vector operator-(const Vector &vec, const T &val)
{
  Vector vec_out = Vector(vec.size());
  thrust::transform(vec.begin(), vec.end(), vec_out.begin(), _1 - val);
  return vec_out;
}


/*!
  \brief Multiply two vectors.
  \param vec The first vector.
  \param vec2 The second vector.
  \return The product of the two vectors.
*/
template <typename Vector>
Vector operator*(const Vector &vec, const Vector &vec2)
{
  assert(vec.size() == vec2.size());
  Vector vec_out = Vector(vec.size());
  thrust::transform(vec.begin(), vec.end(), vec2.begin(), vec_out.begin(), _1 * _2);
  return vec_out;
}

/*!
  \brief Multiply a vector and a scalar.
  \param vec The vector.
  \param val The scalar.
  \return The product of the vector and the scalar.
*/
template <typename T, typename Vector>
Vector operator*(const Vector &vec, const T &val)
{
  Vector vec_out = Vector(vec.size());
  thrust::transform(vec.begin(), vec.end(), vec_out.begin(), _1 * val);
  return vec_out;
}

/*!
  \brief Multiply a scalar and a vector.
  \param val The scalar.
  \param vec The vector.
  \return The product of the vector and the scalar.
*/
template <typename T, typename Vector>
Vector operator*(const T &val, const Vector &vec)
{
  Vector vec_out = Vector(vec.size());
  thrust::transform(vec.begin(), vec.end(), vec_out.begin(), _1 * val);
  return vec_out;
}

/*!
  \brief Divide two vectors.
  \param vec The first vector.
  \param vec2 The second vector.
  \return The quotient of the two vectors.
*/
template <typename Vector>
Vector operator/(const Vector &vec, const Vector &vec2)
{
  assert(vec.size() == vec2.size());
  Vector vec_out = Vector(vec.size());
  thrust::transform(vec.begin(), vec.end(), vec2.begin(), vec_out.begin(), _1 / _2);
  return vec_out;
}

/*!
  \brief Divide a vector by a scalar.
  \param vec The vector.
  \param val The scalar.
  \return The quotient of the vector and the scalar.
*/
template <typename T, typename Vector>
Vector operator/(const Vector &vec, const T &val)
{
  Vector vec_out = Vector(vec.size());
  thrust::transform(vec.begin(), vec.end(), vec_out.begin(), _1 / val);
  return vec_out;
}



} // end namespace arithmetic

} // end namespace common

NMRECON_NAMESPACE_END

#endif // !NMRECON_ARITHMETIC_H_