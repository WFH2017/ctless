
#include <thrust/fill.h>
#include "cuda/image3d.h"
#include "cuda/device_arithmetic.h"

NMRECON_NAMESPACE_BEGIN

namespace common
{
  using namespace nmrecon::common::device_arithmetic;
  // DeviceImage3D function implementations

  __device__ __host__ precision_t DeviceImage3D::get_value_by_index(const size_t &ix, const size_t &iy, const size_t &iz) const
  {
    return data[ix + iy * coordinate.size.x + iz * coordinate.size.x * coordinate.size.y];
  }

  __device__ void DeviceImage3D::set_value_by_index(const size_t &ix, const size_t &iy, const size_t &iz, const precision_t &value)
  {
    atomicAdd(data + (ix + iy * coordinate.size.x + iz * coordinate.size.x * coordinate.size.y), value);
  }

  __device__ __host__ precision_t DeviceImage3D::get_pos_x_by_index(const size_t &ix) const
  {
    return coordinate.origin.x + ix * coordinate.spacing.x;
  }

  __device__ __host__ precision_t DeviceImage3D::get_pos_y_by_index(const size_t &iy) const
  {
    return coordinate.origin.y + iy * coordinate.spacing.y;
  }

  __device__ __host__ precision_t DeviceImage3D::get_pos_z_by_index(const size_t &iz) const
  {
    return coordinate.origin.z + iz * coordinate.spacing.z;
  }

  __device__ __host__ Vec3 DeviceImage3D::get_pos_by_index(const size_t &ix, const size_t &iy, const size_t &iz) const
  {
    return Vec3(
        get_pos_x_by_index(ix),
        get_pos_y_by_index(iy),
        get_pos_z_by_index(iz));
  }

  __device__ __host__ int DeviceImage3D::locate_index_x_by_pos(const precision_t &x) const
  {
    precision_t spacing = coordinate.spacing.x;
    return (x + spacing / 2 - coordinate.origin.x) / spacing;
  }

  __device__ __host__ int DeviceImage3D::locate_index_y_by_pos(const precision_t &y) const
  {
    precision_t spacing = coordinate.spacing.y;
    return (int)(y + spacing / 2 - coordinate.origin.y) / spacing;
  }

  __device__ __host__ int DeviceImage3D::locate_index_z_by_pos(const precision_t &z) const
  {
    precision_t spacing = coordinate.spacing.z;
    return (int)(z + spacing / 2 - coordinate.origin.z) / spacing;
  }

  __device__ __host__ IndexInt DeviceImage3D::locate_index_by_pos(const precision_t &x,
                                                                  const precision_t &y,
                                                                  const precision_t &z) const
  {
    return IndexInt(
        locate_index_x_by_pos(x),
        locate_index_y_by_pos(y),
        locate_index_z_by_pos(z));
  }

  __device__ __host__ Coordinate DeviceImage3D::get_coordinate() const
  {
    return coordinate;
  }

  // Image3D function implementations

  Image3D::Image3D(const Image3D &image3d) : coordinate(image3d.coordinate), data(image3d.data) {}

  Image3D::Image3D(const Coordinate &coordinate_) : coordinate(coordinate_)
  {
    size_t num_voxels = coordinate.size.product();
    data = DeviceVector(num_voxels, 0.0);
  }

  Image3D::Image3D(const Coordinate &coordinate_, const DeviceVector &data_) : coordinate(coordinate_), data(data_) {}

  Image3D& Image3D::operator=(const Image3D &image3d)
  {
    if (this != &image3d)
    {
      
    coordinate = image3d.coordinate;
    data = image3d.data;
    }
    return *this;
  }

  DeviceImage3D Image3D::to_device() 
  {
    return {coordinate, thrust::raw_pointer_cast(data.data())};
  }

  DeviceVector Image3D::get_data() const
  {
    return data;
  }

  void Image3D::set_data(const DeviceVector &data_)
  {
    data = data_;
  }

  Image3D Image3D::operator+(const Image3D &image3d) const
  {
    DeviceVector result_data = data + image3d.data;
    Image3D result(this->coordinate, result_data);
    return result;
  }

  Image3D Image3D::operator+(const precision_t &value) const
  {
    DeviceVector result_data = data + value;
    Image3D result(this->coordinate, result_data);
    return result;
  }

  Image3D Image3D::operator-(const Image3D &image3d) const
  {
    DeviceVector result_data = data - image3d.data;
    Image3D result(this->coordinate, result_data);
    return result;
  }

  Image3D Image3D::operator-(const precision_t &value) const
  {
    DeviceVector result_data = data - value;
    Image3D result(this->coordinate, result_data);
    return result;
  }

  Image3D Image3D::operator*(const Image3D &image3d) const
  {
    DeviceVector result_data = data * image3d.data;
    Image3D result(this->coordinate, result_data);
    return result;
  }

  Image3D Image3D::operator*(const precision_t &value) const
  {
    DeviceVector result_data = data * value;
    Image3D result(this->coordinate, result_data);
    return result;
  }

  Image3D Image3D::operator/(const Image3D &image3d) const
  {
    DeviceVector result_data = true_div(data, image3d.data);
    Image3D result(this->coordinate, result_data);
    return result;
  }

  Image3D Image3D::operator/(const precision_t &value) const
  {
    DeviceVector result_data = data / value;
    Image3D result(this->coordinate, result_data);
    return result;
  }

  void Image3D::fill(const precision_t &value)
  {
    thrust::fill(data.begin(), data.end(), value);
  }

  void Image3D::set_value_by_index(const size_t &ix, const size_t &iy, const size_t &iz, const precision_t &value)
  {
    data[ix + iy * coordinate.size.x + iz * coordinate.size.x * coordinate.size.y] = value;
  }

  precision_t Image3D::get_value_by_index(const size_t &ix, const size_t &iy, const size_t &iz) const
  {
    return data[ix + iy * coordinate.size.x + iz * coordinate.size.x * coordinate.size.y];
  }

  precision_t Image3D::get_value_by_index(const size_t &ind) const
  {
    return data[ind];
  }

} // namespace common

NMRECON_NAMESPACE_END
