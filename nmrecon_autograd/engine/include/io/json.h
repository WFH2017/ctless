/**
 * @file json.h
 * @brief JSON file IO
 */

#pragma once
#include <nlohmann/json.hpp>
#include "common/namespace.h"

NMRECON_NAMESPACE_BEGIN
namespace io
{
    using json = nlohmann::json;
    
} // namespace io

NMRECON_NAMESPACE_END
