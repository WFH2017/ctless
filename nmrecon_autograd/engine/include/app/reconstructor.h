/*
Copyright (C) 2023, Vplus Technology Co., Ltd.
This file is part of NMRecon.
See NMRecon/LICENSE.txt for details.
*/

#pragma once

#include <iostream>

#include "app/base.h"
#include "common/namespace.h"
#include "cuda/reconstruction/em.h"
#include "io/reconstruct_setting.h"
#include "io/scan_info.h"

NMRECON_NAMESPACE_BEGIN
namespace app
{

    class Reconstructor : public BaseApp
    {
    public:
        // Parses the configuration file
        int parse(const std::string &config_file) override;

        // Runs the reconstruction process
        int run() const override;

        // Parses the configuration file from backend, which is used for software call.
        int parse_from_backend(const std::string &config_file);
        // Runs the reconstruction process from backend, which is used for software call.
        int run_from_backend() const;

        nmrecon::common::Coordinate3D compute_recon_fov() const;

        // specifies the recon fov depends on the manufacturer's setting.
        // it is different for different manufacturers.
        std::vector<precision_t> compute_origin_offset() const;

        int run_from_nifty() const;

        // Forward-project a dose-calibrated NIfTI reconstruction with the
        // same DistanceDriven + CT attenuation + CDR operator used by the
        // NIfTI reconstruction path.  The output is float raw-count space.
        int forward_project_from_nifty(const std::string &image_path,
                                       const std::string &output_path) const;

        // Forward-project raw float32 volume without uint16 quantization or dose division
        int forward_project_from_rawdata(const std::string &image_bin_path,
                                         const std::string &output_bin_path) const;

        // Back-project raw projection counts into image domain with the
        // exact same DistanceDriven + CT attenuation + CDR adjoint operator.
        int backproject_from_rawdata(const std::string &projection_bin_path,
                                     const std::string &output_bin_path) const;

        // Export the identically preprocessed main-window and additive
        // scatter projections that are used by the NIfTI OSEM path.
        int export_osem_measurement_from_nifty(const std::string &main_output_path,
                                               const std::string &scatter_output_path) const;

        // Runtime overrides for controlled reconstruction sensitivity tests.
        void set_num_iterations(int value);
        void set_output_path(const std::string &value);
        void disable_cdr();

        void set_mu_map_cm(const std::string &path) { mu_map_cm_path = path; }
        void set_export_mu_map_cm(const std::string &path) { export_mu_map_cm_path = path; }

        // Prints help information
        void help() const override;

    private:
        std::string mu_map_cm_path;
        std::string export_mu_map_cm_path;
        nmrecon::io::ScanInfo scan_info;
        nmrecon::io::ReconstructSetting reconstruct_setting;
    };

} // namespace app
NMRECON_NAMESPACE_END
