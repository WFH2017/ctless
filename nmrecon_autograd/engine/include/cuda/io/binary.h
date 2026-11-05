#pragma once

#include "io/common.h"

NMRECON_NAMESPACE_BEGIN
namespace io
{

    void save_bin(const char *filepath, Image &image);

    nmrecon::common::DeviceVector load_image(const char *filepath, Image &image);

    void save_rawdata(const char *filepath, ProjectionValue &projection);

    void load_rawdata(const char *filepath, ProjectionValue &projection);

    void load_rawdata_host(const char *filepath, thrust::host_vector<float> &h_projection);

    void convert_and_load_rawdata(const char *filepath, ProjectionValue &projection);
}
NMRECON_NAMESPACE_END
