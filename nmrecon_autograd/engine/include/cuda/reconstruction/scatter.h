#pragma once

#include "common/config.h"
#include "common/basic_type.h"
#include "cuda/reconstruction/projection.h"

NMRECON_NAMESPACE_BEGIN
namespace reconstruction
{
    namespace correction
    {
        namespace scatter
        {
            bool smooth_projection(nmrecon::reconstruction::rawdata::Projection &projection, 
                float cutoff, int order);

        }
    }

}
NMRECON_NAMESPACE_END