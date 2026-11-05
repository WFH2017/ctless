#pragma once

#include "common/config.h"
#include "common/basic_type.h"

NMRECON_NAMESPACE_BEGIN

namespace common
{
  
  typedef nmrecon::common::Coordinate3D Coordinate;
  typedef nmrecon::common::Value3<size_t> Index;
  typedef nmrecon::common::Value3<int> IndexInt;
  

  struct DeviceImage3D
  {
    Coordinate coordinate;
    precision_t *data;

    __device__ __host__  precision_t get_value_by_index(const size_t &ix, const size_t &iy, const size_t &iz) const;
    __device__ void set_value_by_index(const size_t &ix, const size_t &iy, const size_t &iz, const precision_t &value);
    __device__ __host__ precision_t get_pos_x_by_index(const size_t &ix) const;
    __device__ __host__ precision_t get_pos_y_by_index(const size_t &iy) const;
    __device__ __host__ precision_t get_pos_z_by_index(const size_t &iz) const;
    __device__ __host__ Vec3 get_pos_by_index(const size_t &ix, const size_t &iy, const size_t &iz) const;
    __device__ __host__ int locate_index_x_by_pos(const precision_t &x) const;
    __device__ __host__ int locate_index_y_by_pos(const precision_t &y) const;
    __device__ __host__ int locate_index_z_by_pos(const precision_t &z) const;
    __device__ __host__ IndexInt locate_index_by_pos(const precision_t &x, const precision_t &y, const precision_t &z) const;
    __device__ __host__ Coordinate get_coordinate() const;
  };

  class Image3D
  {
  public:
    Image3D(const Image3D &image3d);
    Image3D(const Coordinate &coordinate_);
    Image3D(const Coordinate &coordinate_, const DeviceVector &data_);
    Image3D& operator=(const Image3D &image3d);

    __device__ __host__ const Coordinate &get_coordinate() const {return coordinate;}
    __device__ __host__ const Origin &get_origin() const {return coordinate.origin;}
    __device__ __host__ const Size &get_size() const {return coordinate.size;}
    __device__ __host__ const Spacing &get_spacing() const {return coordinate.spacing;}
    __device__ __host__ const Direction &get_direction() const {return coordinate.direction;}

    DeviceImage3D to_device();
    DeviceVector get_data() const;
    DeviceVector &get_data_ref() { return data; }
    const DeviceVector &get_data_ref() const { return data; }
    void set_data(const DeviceVector &data_);

    Image3D operator+(const Image3D &image3d) const;
    Image3D operator+(const precision_t &value) const;
    Image3D operator-(const Image3D &image3d) const;
    Image3D operator-(const precision_t &value) const;
    Image3D operator*(const Image3D &image3d) const;
    Image3D operator*(const precision_t &value) const;
    Image3D operator/(const Image3D &image3d) const;
    Image3D operator/(const precision_t &value) const;

    void fill(const precision_t &value);
    void set_value_by_index(const size_t &ix, const size_t &iy, const size_t &iz, const precision_t &value);
    precision_t get_value_by_index(const size_t &ix, const size_t &iy, const size_t &iz) const;
    precision_t get_value_by_index(const size_t &ind) const;

  private:
    Coordinate coordinate;
    DeviceVector data;
  };

} // namespace common

NMRECON_NAMESPACE_END
