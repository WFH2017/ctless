#pragma once

#include "common/namespace.h"
#include "common/basic_type.h"
#include "cuda/reconstruction/projection.h"
#include "cuda/io/binary.h"
#include "io/common.h"

NMRECON_NAMESPACE_BEGIN

namespace io
{

     void statistic(ProjectionValue &proj_value, const std::string &name);

     void load_from_binary(const std::string &filepath,
                           ProjectionValue &projection);

#ifndef NO_DCMTK
     void load_from_dicom(DcmFileFormat &raw_dcm,
                          ProjectionValue &projection_value,
                          ProjectionValue &scatter_value);
#endif

     void load_from_nifty(const nmrecon::io::ScanInfo &scan_info,
                          ProjectionValue &proj_value,
                          ProjectionValue &scatter_value);

     void copy_from_pinned_to_gpu(const HostPinnedBuffer &buffer,
                                  ProjectionValue &proj_value,
                                  ProjectionValue &scatter_value);

     nmrecon::common::Image3D dose_calibrated(const Image &image_in, precision_t dose_calibration_factor = 200.0);
     nmrecon::common::DeviceVector post_filtering(const Image &image_in, precision_t gaussian_sigma);

     // 将GPU中的float数组转化为CPU中的uint16数组
     std::vector<uint16_t> convertFloatDeviceToUint16Host(const nmrecon::common::DeviceVector &d_input);

}

NMRECON_NAMESPACE_END
