#pragma once
#include "common/namespace.h"
#include "io/json.h"

#include "io/reconstruction.h"
#include "io/correction.h"
#include "io/output.h"

NMRECON_NAMESPACE_BEGIN
namespace io
{

    struct ReconstructSetting
    {
        Reconstruction reconstruction;
        Corrections corrections;
        Output output;
        void from_json(const json &j);
        json to_json() const;
        void display() const;
        bool validate() const;
        void set_default();

    };

} // namespace io
NMRECON_NAMESPACE_END
