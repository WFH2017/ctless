#pragma once

#include <iostream>
#include "common/config.h"

NMRECON_NAMESPACE_BEGIN
namespace common
{

  /**
   * @brief Array with fixed size and can be created on both host and device.
   * the container can be changed by template parameter Cont: default is thrust::host_vector,
   * and be changed to thrust::device_vector.
   * TODO: Array can be derived from Vector, but now I can not figure out how to do it.
   * @tparam T 
   * @tparam Size 
   * @tparam Cont 
   */
  template <typename T, std::size_t Size, 
  template<typename Elem, typename Alloc = std::allocator<Elem>> class Cont=thrust::host_vector>
  class Array 
  {
    // typedef typename nmrecon::common::Vector<T, Cont> Base;
    // typedef typename Base::Data Data;
    typedef Cont<T> Data;
  public:
    // Constructor: initialize the array
    Array(){
      this->data = Data(Size);
    }
    Array(const Array &other):data(other.data){}
    Array(const Data &data_)
    {
      this->data = Data(Size);
      thrust::copy(data_.begin(), data_.begin() + Size, this->data.begin());
    }
    Array &operator=(const Array & other)
    {
      this->data = other.data;
      return *this;
    }
    Array &operator=(const Data & data_)
    {
      thrust::copy(data_.begin(), data_.begin() + Size, data.begin());
      return *this;
    }

    // Get the size of the array
    constexpr size_t size() const
    {
      return Size;
    }

    // Get the value of an element at a specific index (const)
    const T &operator[](size_t index) const
    {
      if (index >= Size)
      {
        throw std::out_of_range("Array index out of range");
      }
      return data[index];
    }

    // set the value of an element at a specific index
    void set(const size_t index, const T& value)
    {
      if (index >= Size)
      {
        throw std::out_of_range("Array index out of range");
      }
      data[index] = value;
    }

    // Get data
    const Data &get_data() const
    {
      return this->data;
    }
    void set_data(const Data &data_)
    {
      thrust::copy(data_.begin(), data_.begin() + Size, this->data.begin());
    }

    T product() const
    {
      for (int i = 0; i < Size; i++)
      {
        std::cout<< data[i] << " ";
      }
      return thrust::reduce(data.begin(), data.end(), 1, thrust::multiplies<T>());
    }

    T sum () const
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
      for (size_t i = 0; i < Size; i++)
      {
        std::cout << data[i] << " ";
      }
      std::cout << std::endl;
    }




  private:
    Data data;
  };

} // end namespace common

NMRECON_NAMESPACE_END
