#pragma once
#ifndef NMRECON_VECTOR_H_
#define NMRECON_VECTOR_H_

#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>

#include "common/config.h"

NMRECON_NAMESPACE_BEGIN

namespace common
{
  template <typename T,
            template <typename Elem, typename Alloc = std::allocator<Elem>> class Cont = thrust::host_vector>
  class Vector
  {
  public:
    typedef Cont<T> Data;
    Vector() {}
    Vector(const Vector &vec) : data(vec.data) {}
    Vector(const Data &data_) : data(data_) {}
    Vector &operator=(const Vector &vec)
    {
      this->data = vec.data;
      return *this;
    }
    Vector &operator=(const Data &data_)
    {
      this->data = data_;
      return *this;
    }

    const T &operator[](const size_t index) const
    {
      if (index >= data.size())
      {
        throw std::out_of_range("Vector::operator[]");
      }
      return data[index];
    }
    constexpr size_t size() const { return data.size(); }

    void set(const size_t index, const T &value)
    {
      if (index >= data.size())
      {
        throw std::out_of_range("Vector::set");
      }
      data[index] = value;
    }

    const Data &get_data() const { return data; }

    void set_data(const Data &data_)
    {
      data = data_;
    }

    T product() const
    {
      return thrust::reduce(data.begin(), data.end(), 1, thrust::multiplies<T>());
    }

    T sum() const
    {
      return thrust::reduce(data.begin(), data.end(), 0, thrust::plus<T>());
    }

    T norm() const
    {
      return sqrt(thrust::inner_product(data.begin(), data.end(), data.begin(), 0.0));
    }

    T max() const
    {
      return *thrust::max_element(data.begin(), data.end());
    }
    T min() const
    {
      return *thrust::min_element(data.begin(), data.end());
    }

    void print() const
    {
      for (size_t i = 0; i < data.size(); i++)
      {
        std::cout << data[i] << " ";
      }
      std::cout << std::endl;
    }

  private:
    Data data;

  }; // class Vector

} // namespace common

NMRECON_NAMESPACE_END

#endif // NMRECON_VECTOR_H_