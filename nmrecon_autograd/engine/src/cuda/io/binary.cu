#include <iostream>
#include <fstream>
#include "cuda/io/binary.h"
NMRECON_NAMESPACE_BEGIN
namespace io
{
  void save_bin(const char *filepath, Image &image)
  {
    thrust::host_vector<float> h_image = image.get_data();
    auto raw_ptr = thrust::raw_pointer_cast(h_image.data());
    std::ofstream ofs(filepath, std::ios::binary | std::ios::out);
    ofs.write((const char *)raw_ptr, image.get_size().product() * sizeof(float));
    ofs.close();
  }

  nmrecon::common::DeviceVector load_image(const char *filepath, Image &image)
  {
    // const struct an image on host.
    thrust::host_vector<float> h_image(image.get_size().product(), 0.0f);
    auto raw_ptr = thrust::raw_pointer_cast(h_image.data());
    std::ifstream ifs(filepath, std::ios::binary | std::ios::in);
    ifs.read((char *)raw_ptr, image.get_size().product() * sizeof(float));
    ifs.close();
    // transfer data from host to device.
    nmrecon::common::DeviceVector d_image = h_image;
    return d_image;
  }

  void save_rawdata(const char *filepath, ProjectionValue &projection)
  {
    thrust::host_vector<float> h_projection = projection;
    auto raw_ptr = thrust::raw_pointer_cast(h_projection.data());
    std::ofstream ofs(filepath, std::ios::binary | std::ios::out);
    ofs.write((const char *)raw_ptr, projection.size() * sizeof(float));
    ofs.close();
  }

  void load_rawdata(const char *filepath, ProjectionValue &projection)
  {
    thrust::host_vector<float> h_projection(projection.size(), 0.0f);
    auto raw_ptr = thrust::raw_pointer_cast(h_projection.data());
    std::ifstream ifs(filepath, std::ios::binary | std::ios::in);
    // Check if file is open
    if (!ifs.is_open())
    {
      std::cerr << "File not found: " << filepath << std::endl;
      std::abort();
    }
    // Read data from file
    ifs.read((char *)raw_ptr, projection.size() * sizeof(float));
    ifs.close();

    thrust::copy(h_projection.begin(), h_projection.end(), projection.begin());
  }

  void load_rawdata_host(const char *filepath, thrust::host_vector<float> &h_projection)
  {
    auto raw_ptr = thrust::raw_pointer_cast(h_projection.data());
    std::ifstream ifs(filepath, std::ios::binary | std::ios::in);
    // Check if file is open
    if (!ifs.is_open())
    {
      std::cerr << "File not found: " << filepath << std::endl;
      std::abort();
    }
    // Read data from file
    ifs.read((char *)raw_ptr, h_projection.size() * sizeof(float));
    ifs.close();
  }

  void convert_and_load_rawdata(const char *filepath, ProjectionValue &projection)
  {
    // thrust::host_vector<u_int16_t>
    nmrecon::h_vec_u16 h_projection_int(projection.size(), 0);
    auto raw_ptr = thrust::raw_pointer_cast(h_projection_int.data());
    std::ifstream ifs(filepath, std::ios::binary | std::ios::in);
    // Check if file is open
    if (!ifs.is_open())
    {
      std::cerr << "File not found: " << filepath << std::endl;
      std::abort();
    }
    // Read data from file
    ifs.read((char *)raw_ptr, projection.size() * sizeof(uint16_t));
    ifs.close();

    // Convert int data to float
    nmrecon::h_vec_f32 h_projection(projection.size(), 0.0f);
    for (size_t i = 0; i < h_projection.size(); ++i)
    {
      h_projection[i] = static_cast<float>(h_projection_int[i]);
    }

    thrust::copy(h_projection.begin(), h_projection.end(), projection.begin());
  }
}
NMRECON_NAMESPACE_END