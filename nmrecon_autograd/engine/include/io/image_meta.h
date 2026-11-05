
#pragma once

#include <vector>
#include <iostream>
#include <nlohmann/json.hpp>
#include <common/namespace.h>
#include <common/types.h>

using json = nlohmann::json;

NMRECON_NAMESPACE_BEGIN
namespace io
{
    struct ImageMeta
    {
        std::vector<size_t> size;
        std::vector<precision_t> spacing;
        std::vector<precision_t> origin;
        std::vector<precision_t> direction;

        void from_json(const json &j);
        json to_json() const;
        void display() const;
        bool validate() const;
        void set_default();
    };

} // namespace io

NMRECON_NAMESPACE_END
