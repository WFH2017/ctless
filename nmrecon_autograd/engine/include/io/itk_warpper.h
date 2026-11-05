#pragma once
#include <vector>

#include <string>
#include <cstdint>

#include "common/namespace.h"
NMRECON_NAMESPACE_BEGIN

namespace io
{

    std::vector<uint16_t> load_spect_nifty_cpu(const std::string& path);


}


NMRECON_NAMESPACE_END