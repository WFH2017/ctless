/*
Copyright (C) 2023, Vplus Technology Co., Ltd.
This file is part of NMRecon.
See NMRecon/LICENSE.txt for details.
*/

#include <fstream>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <string>
#include <algorithm> // For std::transform
#include <cctype>    // For ::tolower

#include "app/reconstructor.h"
#include "scanner/spect.h"

#include "io/dicom.h"
#include "io/itk.h"

#include "cuda/io/binary.h"
#include "cuda/reconstruction/cdr.h"
#include "cuda/reconstruction/attenuation.h"
#include "cuda/reconstruction/distance_driven.h"
#include "cuda/reconstruction/projection.h"
#include "cuda/filter.h"
#include "cuda/io/utils.h"
#include "cuda/device_arithmetic.h"
#include "cuda/io/io.h"

#include <nlohmann/json.hpp>
#include <boost/filesystem.hpp>

using json = nlohmann::json;

using nmrecon::reconstruction::correction::cdr::GaussianCDR;
typedef nmrecon::common::Coordinate3D Coordinate;
typedef nmrecon::io::ImageMeta CoordinateInfo;
typedef nmrecon::common::Value3<float> F3;
typedef nmrecon::common::Value3<size_t> S3;
typedef nmrecon::common::Matrix3x3 Mat3;

NMRECON_NAMESPACE_BEGIN
namespace app
{

    using nmrecon::reconstruction::algorithm::common_recon;
    using nmrecon::reconstruction::algorithm::osem;
    using nmrecon::reconstruction::algorithm::ramla;
    using nmrecon::reconstruction::rawdata::Projection;

    namespace fs = boost::filesystem; // 使用命名空间别名，简化代码

    std::ifstream check_and_read_file(const std::string &filename)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            std::cerr << "Error opening file: " << filename << std::endl;
            std::exit(EXIT_FAILURE);
        }
        return file;
    }

    int make_output_dir(const std::string &out_filename)
    {
        if (out_filename.empty())
        {
            return -1; // 或抛出异常
        }

        fs::path output_path(out_filename);
        fs::path output_dir = output_path.parent_path();

        if (output_dir.empty())
        {
            return 0; // 文件在当前目录
        }

        try
        {
            if (fs::create_directories(output_dir))
            {
                return 0; // 成功创建目录
            }
            else
            {
                // create_directories 返回 false 有多种可能
                if (fs::exists(output_dir))
                {
                    return 0; // 目录已经存在
                }
                std::cerr << "Failed to create directory: " << output_dir.string() << std::endl; // 使用 .string() 转换为 std::string
                return -1;
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception caught while creating directory: " << e.what() << std::endl;
            return -1;
        }
    }

    Image create_image(CoordinateInfo coord_info, float value = 0.0)
    {

        Coordinate coord;
        coord.set_size(S3(coord_info.size[0], coord_info.size[1], coord_info.size[2]));
        coord.set_spacing(F3(coord_info.spacing[0], coord_info.spacing[1], coord_info.spacing[2]));
        coord.set_origin(F3(coord_info.origin[0], coord_info.origin[1], coord_info.origin[2]));

        coord.set_direction(Mat3(coord_info.direction[0], coord_info.direction[1], coord_info.direction[2],
                                 coord_info.direction[3], coord_info.direction[4], coord_info.direction[5],
                                 coord_info.direction[6], coord_info.direction[7], coord_info.direction[8]));
        Image img(coord);
        img.fill(value);
        return img;
    }

    // 工具函数：float → uint16_t 的安全转换
    __host__ __device__ inline uint16_t clamp_and_cast(float x)
    {
        return static_cast<uint16_t>(fminf(65535.0f, fmaxf(0.0f, x + 0.5f)));
    }

    int Reconstructor::parse(const std::string &config_file)
    {
        auto file = check_and_read_file(config_file);
        try
        {
            json j;
            file >> j;
            scan_info.from_json(j["ScanInfo"]);
            reconstruct_setting.from_json(j["ReconstructSetting"]);
            std::cout << "Configuration parsed successfully.\n";
            scan_info.display();
            reconstruct_setting.display();
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error while parsing config: " << e.what() << std::endl;
        }
        return 0;
    }

    int Reconstructor::run() const
    {
        // construct objects
        // set scanner and load rawdata
        // TODO: add scanner choice
        auto scanner = nmrecon::scanner::SPECT(scan_info.scanner.setting);
        // scanner.print();
        DcmFileFormat raw_dcm;
        auto projections = nmrecon::io::load_projection(scan_info, reconstruct_setting.corrections.scatter.type, raw_dcm);
        // debug
        std::cout << "Rawdata loaded successfully.\n";

        // create image
        auto emap = create_image(reconstruct_setting.output.image_meta, 0.0f);

        // load atten_map if exists or create a zero-value one.
        auto atten_map = create_image(reconstruct_setting.output.image_meta, 0.0f);
        auto atten_map_path = reconstruct_setting.corrections.attenuation.ct_image;
        if (atten_map_path.has_value)
        {
            auto atten_map_data = nmrecon::io::load_image(atten_map_path.value.c_str(), emap);
            atten_map.set_data(atten_map_data);
            std::cout << "Attenuation map loaded successfully.\n";
        }
        // configure reconstruct method
        bool use_cdr = false;
        auto cdr_paras = reconstruct_setting.corrections.psf.cdr_kernel;
        GaussianCDR cdr_model(cdr_paras[0], cdr_paras[1], cdr_paras[2]);
        if (reconstruct_setting.corrections.psf.type == "cdr")
        {
            use_cdr = true;
            std::cout << "Using CDR correction.\n";
        }
        else
        {
            std::cout << "Not using CDR correction.\n";
        }

        int num_iters = reconstruct_setting.reconstruction.num_iterations;
        int num_subsets = reconstruct_setting.reconstruction.num_subsets;
        auto algorithm = reconstruct_setting.reconstruction.type;

        std::string out_filename = reconstruct_setting.output.path;
        if (make_output_dir(out_filename) != 0)
            return -1;

        // reconstruct
        std::cout << "Starting reconstruction process...\n";

        int save_interval = reconstruct_setting.output.save_interval;

        auto out_file = out_filename + ".bin";
        if (algorithm == "OSEM")
        {
            auto img = osem(scanner, emap, atten_map, projections, num_iters, num_subsets, save_interval, out_filename, use_cdr, cdr_model);
            nmrecon::io::save_bin(out_file.c_str(), img);
        }
        else if (algorithm == "RAMLA")
        {
            auto img = ramla(scanner, emap, atten_map, projections, num_iters, num_subsets, save_interval, out_filename, use_cdr, cdr_model);
            nmrecon::io::save_bin(out_file.c_str(), img);
        }
        else
        {
            std::cerr << "Unknown reconstruction algorithm: " << algorithm << std::endl;
            return -1;
        }

        std::cout << "Reconstruction completed successfully.\n";
        return 0;
    }

    void Reconstructor::help() const
    {
        std::cout << "Usage: \n";
        std::cout << "  --config <config_file> : Specify the configuration file for reconstruction\n";
        std::cout << "  --help : Show this help message\n";
    }

    /**
     *
     */
    int Reconstructor::parse_from_backend(const std::string &config_file)
    {
        auto file = check_and_read_file(config_file);
        try
        {
            json backend_j;
            file >> backend_j;
            // 构造扫描信息,根据 backend_j 中的原始dicom构造扫描信息
            scan_info.from_dicom(backend_j);
            // 构造重建设置，根据 backend_j中信息构造重建配置
            reconstruct_setting.from_json(backend_j["ReconstructSetting"]);

            // TODO:考虑存储json配置
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << '\n';
        }
        return 0;
    }

    bool containsSiemensCaseInsensitive(const std::string &text)
    {
        // 将要搜索的文本转换为小写
        std::string lowerText = text;
        std::transform(lowerText.begin(), lowerText.end(), lowerText.begin(),
                       [](unsigned char c)
                       { return std::tolower(c); }); // 使用 lambda 表达式和 unsigned char 避免 char 签名问题

        // 定义要查找的关键词（小写）
        std::string keyword = "siemens";

        // 使用 find 方法查找
        // find 返回 std::string::npos 如果没有找到，否则返回子串的起始位置
        return lowerText.find(keyword) != std::string::npos;
    }
    bool containsGECaseInsensitive(const std::string &text)
    {
        // 将要搜索的文本转换为小写
        std::string lowerText = text;
        std::transform(lowerText.begin(), lowerText.end(), lowerText.begin(),
                       [](unsigned char c)
                       { return std::tolower(c); }); // 使用 lambda 表达式和 unsigned char 避免 char 签名问题

        // 定义要查找的关键词（小写）
        std::string keyword = "ge";
        return lowerText.find(keyword) != std::string::npos;
    }

    std::vector<precision_t> Reconstructor::compute_origin_offset() const
    {
        std::vector<precision_t> origin_offset = {0.0,0.0,0.0};
        auto manufacturer = scan_info.scanner.type;
        if (containsSiemensCaseInsensitive(manufacturer))
        {
            origin_offset[1] = -scan_info.scanner.setting.table_height-3.26;
            
            origin_offset[2] = -scan_info.scanner.setting.image_position_patient_on_z_axis +
                                    scan_info.scanner.setting.table_traverse;
        }
        else if(containsGECaseInsensitive(manufacturer))
        {
            // origin_offset[0] = scan_info.scanner.setting.image_position_patient_on_z_axis;
        }
        return origin_offset;
    }

    nmrecon::common::Coordinate3D Reconstructor::compute_recon_fov() const
    {
        // 计算投影所对应的FOV 尺寸
        
        precision_t detector_spacing = scan_info.scanner.setting.detector_coordinate.spacing[2];   // 后面两个值，分别表示探测器宽和高,通常相等
        auto origin_modification = compute_origin_offset();

        auto fov_size = reconstruct_setting.output.image_meta.size;
        auto image_meta = reconstruct_setting.output.image_meta;

        precision_t target_spacing;
        if (image_meta.spacing[0] > 0.f)
        {
            target_spacing = image_meta.spacing[0];
        }
        else
        {
            target_spacing = detector_spacing;
        }

        std::vector<precision_t> physical_size;

        for (int i = 0; i < fov_size.size(); i++)
        {
            physical_size.push_back(fov_size[i] * target_spacing);
        }

        auto direction_on_z_axis = scan_info.scanner.setting.image_orientation_patient_on_z_axis; // 1.0 或者 -1.0
        precision_t origin_on_z_axis = scan_info.scanner.setting.image_position_patient_on_z_axis; //
        nmrecon::common::Coordinate3D target_image_meta({target_spacing / 2 - physical_size[0] / 2, target_spacing / 2 - physical_size[1] / 2 + origin_modification[1], origin_on_z_axis },
                                                        {image_meta.size[0], image_meta.size[1], image_meta.size[2]},
                                                        {target_spacing, target_spacing, target_spacing},
                                                        {1, 0, 0, 0, 1, 0, 0, 0, direction_on_z_axis});

        return target_image_meta;
    }
    int Reconstructor::run_from_backend() const
    {

        scan_info.display();
        reconstruct_setting.display();

        
        auto scanner = nmrecon::scanner::SPECT(scan_info.scanner.setting);
        auto scatter_type = reconstruct_setting.corrections.scatter.type;
        DcmFileFormat raw_dcm;
        auto projections = nmrecon::io::load_projection(scan_info, scatter_type, raw_dcm);
        // debug
        std::cout << "Rawdata loaded successfully.\n";

        // create image
        auto emap = create_image(reconstruct_setting.output.image_meta, 0.0f);

        auto output_image_meta = compute_recon_fov();
        std::cout << "target_image_meta:" << std::endl;
        output_image_meta.print();
        nmrecon::io::SpectFovParameters fov_paras = nmrecon::io::ITKProcessor::compute_fov_parameters(output_image_meta);

        // load atten_map if exists or create a zero-value one.
        auto atten_map = create_image(reconstruct_setting.output.image_meta, 0.0f);

        auto ct_path = reconstruct_setting.corrections.attenuation.ct_image;
        if (ct_path.has_value)
        {
            // auto atten_map_data = nmrecon::io::load_image(atten_map_path.value.c_str(),  emap);
            // atten_map.set_data(atten_map_data);

            nmrecon::io::ItkCTImageType::Pointer ct_image = nmrecon::io::ItkCTImageType::New();
            bool load_success = nmrecon::io::ITKProcessor::load_ct_dicom(ct_path.value.c_str(), ct_image);
            if (load_success)
            {
                std::cout << "ct_image loaded successfully.\n";

                // test resample
                std::string output_ct_path_resample = ct_path.value + "_resample.nii.gz";
                nmrecon::io::ItkCTImageType::Pointer resampled_ct = nmrecon::io::ItkCTImageType::New();
                bool resampled_flag = nmrecon::io::ITKProcessor::resample_image_to_fov<nmrecon::io::ItkCTImageType>(ct_image, fov_paras, resampled_ct, static_cast<nmrecon::io::CtPixelType>(-1000));

                if (nmrecon::io::ITKProcessor::save_itk_image<nmrecon::io::ItkCTImageType>(output_ct_path_resample, resampled_ct))
                {
                    std::cout << "resampled_ct saved successfully.\n";
                }
                auto ct_data_vector = nmrecon::io::ITKProcessor::convert_itk_image_to_vector<nmrecon::io::ItkCTImageType>(resampled_ct);
                auto atten_map_data = nmrecon::reconstruction::correction::attenuation::hu_to_mu_map(ct_data_vector);
                atten_map.set_data(atten_map_data);
            }
            else
            {
                std::cerr << "Failed to load ct_image.\n";
            }
        }

        // configure reconstruct method
        bool use_cdr = false;
        auto cdr_paras = reconstruct_setting.corrections.psf.cdr_kernel;
        GaussianCDR cdr_model(cdr_paras[0], cdr_paras[1], cdr_paras[2]);
        if (reconstruct_setting.corrections.psf.type == "cdr")
        {
            use_cdr = true;
            std::cout << "Using CDR correction.\n";
        }
        else
        {
            std::cout << "Not using CDR correction.\n";
        }

        int num_iters = reconstruct_setting.reconstruction.num_iterations;
        int num_subsets = reconstruct_setting.reconstruction.num_subsets;
        int save_interval = reconstruct_setting.output.save_interval;
        auto algorithm = reconstruct_setting.reconstruction.type;

        std::string out_filepath = reconstruct_setting.output.path;
        if (make_output_dir(out_filepath) != 0)
            return -1;

        // reconstruct
        std::cout << "Starting reconstruction process...\n";

        auto final_image = create_image(reconstruct_setting.output.image_meta, 0.0f);

        if (algorithm == "OSEM")
        {
            auto img = osem(scanner, emap, atten_map, projections, num_iters, num_subsets, save_interval, out_filepath, use_cdr, cdr_model);
            final_image = img;
        }
        else if (algorithm == "RAMLA")
        {
            auto img = ramla(scanner, emap, atten_map, projections, num_iters, num_subsets, save_interval, out_filepath, use_cdr, cdr_model);
            final_image = img;
        }
        else
        {
            std::cerr << "Unknown reconstruction algorithm: " << algorithm << std::endl;
            return -1;
        }

        std::cout << "Reconstruction completed successfully.\n";

        // scale the reconstructed image with dose calibration.
        auto scaled_image = nmrecon::io::dose_calibrated(final_image);

        // gaussian filtering
        nmrecon::common::DeviceVector d_float_image = scaled_image.get_data();
        auto filtered_image = nmrecon::io::post_filtering(scaled_image, reconstruct_setting.reconstruction.gaussian_sigma);
        auto out_image = nmrecon::io::convertFloatDeviceToUint16Host(filtered_image);

        // save final image to dicom
        DcmFileFormat out_dcm;
        fs::path fs_path(out_filepath);
        std::string new_sop_instance_uid = fs_path.filename().string();
        std::cout << "out_filepath: " << out_filepath << "," << new_sop_instance_uid << std::endl;
        nmrecon::io::DicomProcessor::transfer_dicom_meta(raw_dcm, out_dcm, new_sop_instance_uid.c_str());

        nmrecon::io::DicomProcessor::set_output_image(out_dcm.getDataset(), out_image, output_image_meta);

        auto out_spect_itk = nmrecon::io::ITKProcessor::convert_vector_to_itk_image<nmrecon::io::ItkNmImageType>(out_image, fov_paras);

        auto itk_spect_filename = out_filepath + ".nii.gz";
        nmrecon::io::ITKProcessor::save_itk_image<nmrecon::io::ItkNmImageType>(itk_spect_filename, out_spect_itk);
        std::cout << "Saved itk image to " << itk_spect_filename << std::endl;

        // save image
        OFString transfer_syntax_uid;
        if (raw_dcm.getMetaInfo()->findAndGetOFString(DCM_TransferSyntaxUID, transfer_syntax_uid).bad())
        {
            std::cerr << "Failed to read Transfer Syntax UID." << std::endl;
            return false;
        }
        E_TransferSyntax xfer = DcmXfer(transfer_syntax_uid.c_str()).getXfer();
        nmrecon::io::DicomProcessor::save_format_to_dicom(out_filepath, out_dcm, xfer);
        return 0;
    }

    int Reconstructor::run_from_nifty() const
    { 
      
        scan_info.display();
        reconstruct_setting.display();
        
        auto scanner = nmrecon::scanner::SPECT(scan_info.scanner.setting);
        auto scatter_type = reconstruct_setting.corrections.scatter.type;

        DcmFileFormat raw_dcm;
        auto projections = nmrecon::io::load_projection(scan_info, scatter_type);
        // debug
        std::cout << "Rawdata loaded successfully.\n";

        // create image
        auto emap = create_image(reconstruct_setting.output.image_meta, 0.0f);

        auto output_image_meta = compute_recon_fov();
        std::cout << "target_image_meta:" << std::endl;
        output_image_meta.print();
        nmrecon::io::SpectFovParameters fov_paras = nmrecon::io::ITKProcessor::compute_fov_parameters(output_image_meta);

        // load atten_map if exists or create a zero-value one.
        auto atten_map = create_image(reconstruct_setting.output.image_meta, 0.0f);

        auto ct_path = reconstruct_setting.corrections.attenuation.ct_image;
        if (ct_path.has_value)
        {
            // auto atten_map_data = nmrecon::io::load_image(atten_map_path.value.c_str(),  emap);
            // atten_map.set_data(atten_map_data);

            nmrecon::io::ItkCTImageType::Pointer ct_image = nmrecon::io::ItkCTImageType::New();
            bool load_success = nmrecon::io::ITKProcessor::load_ct_nifty(ct_path.value.c_str(), ct_image);
            if (load_success)
            {
                std::cout << "ct_image loaded successfully.\n";

                // test resample
                std::string output_ct_path_resample = ct_path.value + "_resample.nii.gz";
                nmrecon::io::ItkCTImageType::Pointer resampled_ct = nmrecon::io::ItkCTImageType::New();
                bool resampled_flag = nmrecon::io::ITKProcessor::resample_image_to_fov<nmrecon::io::ItkCTImageType>(ct_image, fov_paras, resampled_ct, static_cast<nmrecon::io::CtPixelType>(-1000));

                if (nmrecon::io::ITKProcessor::save_itk_image<nmrecon::io::ItkCTImageType>(output_ct_path_resample, resampled_ct))
                {
                    std::cout << "resampled_ct saved successfully.\n";
                }
                auto ct_data_vector = nmrecon::io::ITKProcessor::convert_itk_image_to_vector<nmrecon::io::ItkCTImageType>(resampled_ct);
                auto atten_map_data = nmrecon::reconstruction::correction::attenuation::hu_to_mu_map(ct_data_vector);
                atten_map.set_data(atten_map_data);
            }
            else
            {
                std::cerr << "Failed to load ct_image.\n";
            }
        }

        // configure reconstruct method
        bool use_cdr = false;
        auto cdr_paras = reconstruct_setting.corrections.psf.cdr_kernel;
        GaussianCDR cdr_model(cdr_paras[0], cdr_paras[1], cdr_paras[2]);
        if (reconstruct_setting.corrections.psf.type == "cdr")
        {
            use_cdr = true;
            std::cout << "Using CDR correction.\n";
        }
        else
        {
            std::cout << "Not using CDR correction.\n";
        }

        int num_iters = reconstruct_setting.reconstruction.num_iterations;
        int num_subsets = reconstruct_setting.reconstruction.num_subsets;
        int save_interval = reconstruct_setting.output.save_interval;
        auto algorithm = reconstruct_setting.reconstruction.type;

        std::string out_filepath = reconstruct_setting.output.path;
        if (make_output_dir(out_filepath) != 0)
            return -1;

        // reconstruct
        std::cout << "Starting reconstruction process...\n";

        auto final_image = create_image(reconstruct_setting.output.image_meta, 0.0f);

        if (algorithm == "OSEM")
        {
            auto img = osem(scanner, emap, atten_map, projections, num_iters, num_subsets, save_interval, out_filepath, use_cdr, cdr_model);
            final_image = img;
        }
        else if (algorithm == "RAMLA")
        {
            auto img = ramla(scanner, emap, atten_map, projections, num_iters, num_subsets, save_interval, out_filepath, use_cdr, cdr_model);
            final_image = img;
        }
        else
        {
            std::cerr << "Unknown reconstruction algorithm: " << algorithm << std::endl;
            return -1;
        }

        std::cout << "Reconstruction completed successfully.\n";

        // scale the reconstructed image with dose calibration.
        auto scaled_image = nmrecon::io::dose_calibrated(final_image);

        // gaussian filtering
        nmrecon::common::DeviceVector d_float_image = scaled_image.get_data();
        auto filtered_image = nmrecon::io::post_filtering(scaled_image, reconstruct_setting.reconstruction.gaussian_sigma);
        auto out_image = nmrecon::io::convertFloatDeviceToUint16Host(filtered_image);

        // save final image to dicom
        DcmFileFormat out_dcm;
        fs::path fs_path(out_filepath);
        std::string new_sop_instance_uid = fs_path.filename().string();
        std::cout << "out_filepath: " << out_filepath << "," << new_sop_instance_uid << std::endl;
        nmrecon::io::DicomProcessor::transfer_dicom_meta(raw_dcm, out_dcm, new_sop_instance_uid.c_str());

        nmrecon::io::DicomProcessor::set_output_image(out_dcm.getDataset(), out_image, output_image_meta);

        auto out_spect_itk = nmrecon::io::ITKProcessor::convert_vector_to_itk_image<nmrecon::io::ItkNmImageType>(out_image, fov_paras);

        auto itk_spect_filename = out_filepath + ".nii.gz";
        nmrecon::io::ITKProcessor::save_itk_image<nmrecon::io::ItkNmImageType>(itk_spect_filename, out_spect_itk);
        std::cout << "Saved itk image to " << itk_spect_filename << std::endl;
        return 0;
    }

    int Reconstructor::forward_project_from_nifty(
        const std::string &image_path, const std::string &output_path) const
    {
        auto scanner = nmrecon::scanner::SPECT(scan_info.scanner.setting);
        auto image = create_image(reconstruct_setting.output.image_meta, 0.0f);
        image.set_data(nmrecon::io::load_spect_nifty_image(image_path));

        // run_from_nifty writes the reconstructed activity after a fixed
        // dose_calibrated(..., 200) operation.  Invert that output-only scale
        // before comparing the forward projection with acquired counts.
        image.set_data(nmrecon::common::device_arithmetic::operator/(image.get_data(), 200.0f));

        auto atten_map = create_image(reconstruct_setting.output.image_meta, 0.0f);
        auto output_image_meta = compute_recon_fov();
        nmrecon::io::SpectFovParameters fov_paras =
            nmrecon::io::ITKProcessor::compute_fov_parameters(output_image_meta);
        auto ct_path = reconstruct_setting.corrections.attenuation.ct_image;
        if (ct_path.has_value)
        {
            std::string resampled_path = ct_path.value;
            if (resampled_path.find("_resample.nii.gz") == std::string::npos && fs::exists(ct_path.value + "_resample.nii.gz"))
            {
                resampled_path = ct_path.value + "_resample.nii.gz";
            }
            nmrecon::io::ItkCTImageType::Pointer resampled_ct = nmrecon::io::ItkCTImageType::New();
            if (resampled_path.find("_resample.nii.gz") != std::string::npos && nmrecon::io::ITKProcessor::load_ct_nifty(resampled_path.c_str(), resampled_ct))
            {
                auto ct_data = nmrecon::io::ITKProcessor::convert_itk_image_to_vector<nmrecon::io::ItkCTImageType>(resampled_ct);
                atten_map.set_data(nmrecon::reconstruction::correction::attenuation::hu_to_mu_map(ct_data));
            }
            else
            {
                nmrecon::io::ItkCTImageType::Pointer ct_image = nmrecon::io::ItkCTImageType::New();
                if (!nmrecon::io::ITKProcessor::load_ct_nifty(ct_path.value.c_str(), ct_image))
                {
                    std::cerr << "Failed to load CT attenuation image.\n";
                    return -1;
                }
                if (!nmrecon::io::ITKProcessor::resample_image_to_fov<nmrecon::io::ItkCTImageType>(
                        ct_image, fov_paras, resampled_ct, static_cast<nmrecon::io::CtPixelType>(-1000)))
                {
                    std::cerr << "Failed to resample CT attenuation image.\n";
                    return -1;
                }
                auto ct_data = nmrecon::io::ITKProcessor::convert_itk_image_to_vector<nmrecon::io::ItkCTImageType>(resampled_ct);
                atten_map.set_data(nmrecon::reconstruction::correction::attenuation::hu_to_mu_map(ct_data));
            }
        }

        auto collimator = scanner.get_collimator();
        auto detector_coordinate = collimator.get_coordinate_prototype();
        Projection projection(scanner.get_trajectory_list(), detector_coordinate.size);
        auto &projected_data = projection.get_data_ref();
        auto cdr_parameters = reconstruct_setting.corrections.psf.cdr_kernel;
        GaussianCDR cdr_model(cdr_parameters[0], cdr_parameters[1], cdr_parameters[2]);
        if (reconstruct_setting.corrections.psf.type == "cdr")
            nmrecon::reconstruction::projector::distance_driven::cdr_project(
                scanner, cdr_model, image, atten_map, projected_data);
        else
            nmrecon::reconstruction::projector::distance_driven::atten_project(
                scanner, image, atten_map, projected_data);
        nmrecon::io::save_rawdata(output_path.c_str(), projected_data);
        std::cout << "Forward projection written to " << output_path << "\n";
        return 0;
    }

    int Reconstructor::forward_project_from_rawdata(const std::string &image_bin_path,
                                                    const std::string &output_bin_path) const
    {
        auto scanner = nmrecon::scanner::SPECT(scan_info.scanner.setting);
        auto image = create_image(reconstruct_setting.output.image_meta, 0.0f);
        auto &image_data = image.get_data_ref();
        nmrecon::io::load_rawdata(image_bin_path.c_str(), image_data);

        auto atten_map = create_image(reconstruct_setting.output.image_meta, 0.0f);
        auto output_image_meta = compute_recon_fov();
        nmrecon::io::SpectFovParameters fov_paras =
            nmrecon::io::ITKProcessor::compute_fov_parameters(output_image_meta);
        auto ct_path = reconstruct_setting.corrections.attenuation.ct_image;
        if (!mu_map_cm_path.empty()) {
            const size_t count = atten_map.get_data_ref().size();
            std::ifstream input(mu_map_cm_path, std::ios::binary | std::ios::ate);
            if (!input || input.tellg() != static_cast<std::streamoff>(count * sizeof(float)))
                throw std::runtime_error("mu-map file missing or wrong float32 size");
            input.seekg(0);
            std::vector<float> values(count);
            if (!input.read(reinterpret_cast<char*>(values.data()), count * sizeof(float)))
                throw std::runtime_error("mu-map read failed");
            for (auto &v : values) {
                if (!std::isfinite(v) || v < 0.0f)
                    throw std::runtime_error("mu-map must contain finite nonnegative cm^-1 values");
                v *= 0.1f; // cm^-1 -> mm^-1, matching native geometry
            }
            atten_map.set_data(nmrecon::common::DeviceVector(values.begin(), values.end()));
        }
        else if (ct_path.has_value)
        {
            std::string resampled_path = ct_path.value;
            if (resampled_path.find("_resample.nii.gz") == std::string::npos && fs::exists(ct_path.value + "_resample.nii.gz"))
            {
                resampled_path = ct_path.value + "_resample.nii.gz";
            }
            nmrecon::io::ItkCTImageType::Pointer resampled_ct = nmrecon::io::ItkCTImageType::New();
            if (resampled_path.find("_resample.nii.gz") != std::string::npos && nmrecon::io::ITKProcessor::load_ct_nifty(resampled_path.c_str(), resampled_ct))
            {
                auto ct_data = nmrecon::io::ITKProcessor::convert_itk_image_to_vector<nmrecon::io::ItkCTImageType>(resampled_ct);
                atten_map.set_data(nmrecon::reconstruction::correction::attenuation::hu_to_mu_map(ct_data));
            }
            else
            {
                nmrecon::io::ItkCTImageType::Pointer ct_image = nmrecon::io::ItkCTImageType::New();
                if (!nmrecon::io::ITKProcessor::load_ct_nifty(ct_path.value.c_str(), ct_image))
                {
                    std::cerr << "Failed to load CT attenuation image.\n";
                    return -1;
                }
                if (!nmrecon::io::ITKProcessor::resample_image_to_fov<nmrecon::io::ItkCTImageType>(
                        ct_image, fov_paras, resampled_ct, static_cast<nmrecon::io::CtPixelType>(-1000)))
                {
                    std::cerr << "Failed to resample CT attenuation image.\n";
                    return -1;
                }
                auto ct_data = nmrecon::io::ITKProcessor::convert_itk_image_to_vector<nmrecon::io::ItkCTImageType>(resampled_ct);
                atten_map.set_data(nmrecon::reconstruction::correction::attenuation::hu_to_mu_map(ct_data));
            }
        }

        if (!export_mu_map_cm_path.empty()) {
            const auto &device_mu = atten_map.get_data_ref();
            std::vector<float> values(device_mu.size());
            thrust::copy(device_mu.begin(), device_mu.end(), values.begin());
            for (auto &v : values) v *= 10.0f;
            std::ofstream output(export_mu_map_cm_path, std::ios::binary);
            if (!output.write(reinterpret_cast<const char*>(values.data()), values.size()*sizeof(float)))
                throw std::runtime_error("mu-map export failed");
        }
        auto collimator = scanner.get_collimator();
        auto detector_coordinate = collimator.get_coordinate_prototype();
        Projection projection(scanner.get_trajectory_list(), detector_coordinate.size);
        auto &projected_data = projection.get_data_ref();
        auto cdr_parameters = reconstruct_setting.corrections.psf.cdr_kernel;
        GaussianCDR cdr_model(cdr_parameters[0], cdr_parameters[1], cdr_parameters[2]);
        if (reconstruct_setting.corrections.psf.type == "cdr")
            nmrecon::reconstruction::projector::distance_driven::cdr_project(
                scanner, cdr_model, image, atten_map, projected_data);
        else
            nmrecon::reconstruction::projector::distance_driven::atten_project(
                scanner, image, atten_map, projected_data);
        nmrecon::io::save_rawdata(output_bin_path.c_str(), projected_data);
        return 0;
    }

    int Reconstructor::backproject_from_rawdata(const std::string &projection_bin_path,
                                                const std::string &output_bin_path) const
    {
        auto scanner = nmrecon::scanner::SPECT(scan_info.scanner.setting);
        auto image = create_image(reconstruct_setting.output.image_meta, 0.0f);
        auto atten_map = create_image(reconstruct_setting.output.image_meta, 0.0f);
        auto output_image_meta = compute_recon_fov();
        nmrecon::io::SpectFovParameters fov_paras =
            nmrecon::io::ITKProcessor::compute_fov_parameters(output_image_meta);
        auto ct_path = reconstruct_setting.corrections.attenuation.ct_image;
        if (!mu_map_cm_path.empty()) {
            const size_t count = atten_map.get_data_ref().size();
            std::ifstream input(mu_map_cm_path, std::ios::binary | std::ios::ate);
            if (!input || input.tellg() != static_cast<std::streamoff>(count * sizeof(float)))
                throw std::runtime_error("mu-map file missing or wrong float32 size");
            input.seekg(0);
            std::vector<float> values(count);
            if (!input.read(reinterpret_cast<char*>(values.data()), count * sizeof(float)))
                throw std::runtime_error("mu-map read failed");
            for (auto &v : values) {
                if (!std::isfinite(v) || v < 0.0f)
                    throw std::runtime_error("mu-map must contain finite nonnegative cm^-1 values");
                v *= 0.1f; // cm^-1 -> mm^-1, matching native geometry
            }
            atten_map.set_data(nmrecon::common::DeviceVector(values.begin(), values.end()));
        }
        else if (ct_path.has_value)
        {
            std::string resampled_path = ct_path.value;
            if (resampled_path.find("_resample.nii.gz") == std::string::npos && fs::exists(ct_path.value + "_resample.nii.gz"))
            {
                resampled_path = ct_path.value + "_resample.nii.gz";
            }
            nmrecon::io::ItkCTImageType::Pointer resampled_ct = nmrecon::io::ItkCTImageType::New();
            if (resampled_path.find("_resample.nii.gz") != std::string::npos && nmrecon::io::ITKProcessor::load_ct_nifty(resampled_path.c_str(), resampled_ct))
            {
                auto ct_data = nmrecon::io::ITKProcessor::convert_itk_image_to_vector<nmrecon::io::ItkCTImageType>(resampled_ct);
                atten_map.set_data(nmrecon::reconstruction::correction::attenuation::hu_to_mu_map(ct_data));
            }
            else
            {
                nmrecon::io::ItkCTImageType::Pointer ct_image = nmrecon::io::ItkCTImageType::New();
                if (!nmrecon::io::ITKProcessor::load_ct_nifty(ct_path.value.c_str(), ct_image))
                {
                    std::cerr << "Failed to load CT attenuation image.\n";
                    return -1;
                }
                if (!nmrecon::io::ITKProcessor::resample_image_to_fov<nmrecon::io::ItkCTImageType>(
                        ct_image, fov_paras, resampled_ct, static_cast<nmrecon::io::CtPixelType>(-1000)))
                {
                    std::cerr << "Failed to resample CT attenuation image.\n";
                    return -1;
                }
                auto ct_data = nmrecon::io::ITKProcessor::convert_itk_image_to_vector<nmrecon::io::ItkCTImageType>(resampled_ct);
                atten_map.set_data(nmrecon::reconstruction::correction::attenuation::hu_to_mu_map(ct_data));
            }
        }

        if (!export_mu_map_cm_path.empty()) {
            const auto &device_mu = atten_map.get_data_ref();
            std::vector<float> values(device_mu.size());
            thrust::copy(device_mu.begin(), device_mu.end(), values.begin());
            for (auto &v : values) v *= 10.0f;
            std::ofstream output(export_mu_map_cm_path, std::ios::binary);
            if (!output.write(reinterpret_cast<const char*>(values.data()), values.size()*sizeof(float)))
                throw std::runtime_error("mu-map export failed");
        }
        auto collimator = scanner.get_collimator();
        auto detector_coordinate = collimator.get_coordinate_prototype();
        Projection projection(scanner.get_trajectory_list(), detector_coordinate.size);
        auto &projected_data = projection.get_data_ref();
        nmrecon::io::load_rawdata(projection_bin_path.c_str(), projected_data);

        auto cdr_parameters = reconstruct_setting.corrections.psf.cdr_kernel;
        GaussianCDR cdr_model(cdr_parameters[0], cdr_parameters[1], cdr_parameters[2]);
        if (reconstruct_setting.corrections.psf.type == "cdr")
            nmrecon::reconstruction::projector::distance_driven::cdr_backproject(
                scanner, cdr_model, image, atten_map, projected_data);
        else
            nmrecon::reconstruction::projector::distance_driven::atten_backproject(
                scanner, image, atten_map, projected_data);

        nmrecon::io::save_bin(output_bin_path.c_str(), image);
        std::cout << "Backprojection written to " << output_bin_path << "\n";
        return 0;
    }

    int Reconstructor::export_osem_measurement_from_nifty(
        const std::string &main_output_path,
        const std::string &scatter_output_path) const
    {
        auto projections = nmrecon::io::load_projection(
            scan_info, reconstruct_setting.corrections.scatter.type);
        if (projections.empty())
            throw std::runtime_error("No projection data available for export.");

        auto main_data = projections[0].get_data();
        nmrecon::io::save_rawdata(main_output_path.c_str(), main_data);

        if (projections.size() < 2)
            throw std::runtime_error("The NIfTI OSEM path requires a scatter-window projection.");
        auto scatter_data = projections[1].get_data();
        nmrecon::io::save_rawdata(scatter_output_path.c_str(), scatter_data);
        return 0;
    }

    void Reconstructor::set_num_iterations(int value)
    {
        reconstruct_setting.reconstruction.num_iterations = value;
    }

    void Reconstructor::set_output_path(const std::string &value)
    {
        reconstruct_setting.output.path = value;
    }

    void Reconstructor::disable_cdr()
    {
        reconstruct_setting.corrections.psf.type = "none";
    }

} // namespace app
NMRECON_NAMESPACE_END
