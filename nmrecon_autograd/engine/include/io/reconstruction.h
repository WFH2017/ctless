#pragma once

#include "common/config.h"
#include "io/json.h"

NMRECON_NAMESPACE_BEGIN
namespace io
{

    struct Reconstruction
    {
        std::string type;
        int num_iterations;
        int num_subsets;
        std::string projector;
        precision_t gaussian_sigma;

        void from_json(const json &j);

        json to_json() const;

        void display() const;

        bool validate() const;

        void set_default();
    };

} // namespace io
NMRECON_NAMESPACE_END