#pragma once

#include "common/config.h"
#include "scanner/spect.h"
#include "cuda/reconstruction/cdr.h"
#include "cuda/reconstruction/distance_driven.h"
#include "cuda/test_utils.h"
#include "cuda/io/binary.h"
#include "cuda/image3d.h"
#include "cuda/reconstruction/projection.h"

NMRECON_NAMESPACE_BEGIN

namespace reconstruction
{
  namespace algorithm
  {
    using namespace nmrecon::common;
    using namespace nmrecon::scanner;
    using namespace nmrecon::reconstruction::rawdata;
    using nmrecon::reconstruction::correction::cdr::GaussianCDR;

    typedef Image3D Image;
    typedef DeviceVector ProjectionValue;
    typedef nmrecon::scanner::SPECT Scanner;

    // Image mlem(Scanner &scanner, ProjectionValue &proj_value, Image &image);
    Image common_recon(Scanner &scanner,
                       Image &emap,
                       Image &atten_map,
                       ProjectionValue &proj_value,
                       int num_iters,
                       int save_interval,
                       const std::string &out_file_name,
                       const bool use_cdr);

    Image osem(Scanner &scanner,
               Image &emap,
               Image &atten_map,
               std::vector<Projection> &projection,
               int num_iters,
               int num_subsets,
               int save_interval,
               const std::string &out_filename,
               const bool use_cdr,
               GaussianCDR &cdr_model);

    Image ramla(Scanner &scanner,
                Image &emap,
                Image &atten_map,
                std::vector<Projection> &projections,
                int num_iters,
                int num_subsets,
                int save_interval,
                const std::string &out_filename,
                const bool use_cdr,
                GaussianCDR &cdr_model);

    Image mbsrem(Scanner &scanner,
                 Image &emap,
                 Image &atten_map,
                 std::vector<Projection> &projections,
                 int num_iters,
                 int num_subsets,
                 int save_interval,
                 const std::string &out_filename,
                 const bool use_cdr);

  } // end namespace algorithm

} // end namespace reconstruction

NMRECON_NAMESPACE_END
