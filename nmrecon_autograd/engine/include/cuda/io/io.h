#pragma once

#include <vector>
#include "common/namespace.h"
#include "io/common.h"
#include "cuda/reconstruction/projection.h"


NMRECON_NAMESPACE_BEGIN
namespace io
{


    // 从dicom中读取投影数据
    std::vector<nmrecon::reconstruction::rawdata::Projection> load_projection(const nmrecon::io::ScanInfo &scan_info,
                                                                              const std::string &scatter_type,
                                                                              DcmFileFormat &raw_dcm);


    // 从nifty中读取投影数据
    std::vector<nmrecon::reconstruction::rawdata::Projection> load_projection(const nmrecon::io::ScanInfo &scan_info,
                                                                              const std::string &scatter_type);

    // Load a uint16 NIfTI SPECT image and transfer it to CUDA float storage.
    nmrecon::common::DeviceVector load_spect_nifty_image(const std::string &filepath);

}

NMRECON_NAMESPACE_END
