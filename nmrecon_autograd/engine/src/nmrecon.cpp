#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <nlohmann/json.hpp> // 使用 nlohmann/json
#include "app/reconstructor.h"

using json = nlohmann::json;
using namespace std::chrono;
using namespace nmrecon::app;

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <config_path.json> [--format dicom|nifty] [--iterations N] [--output PATH] [--disable-cdr] [--forward-project <image.nii.gz> <output.bin>] [--export-osem-measurement <main.bin> <scatter.bin>]" << std::endl;
        return 1;
    }

    std::string config_path = argv[1];
    std::string raw_format = "dicom"; // 默认值
    std::string forward_image, forward_output, osem_main_output, osem_scatter_output;
    std::string backproject_proj, backproject_output;
    std::string mu_map_cm_path, export_mu_map_cm_path;
    int iterations_override = -1;
    std::string output_override;
    bool cdr_disabled = false;

    // 简单解析 --format 参数
    for (int i = 2; i < argc; ++i)
    {
        std::string arg = argv[i];
        if ((arg == "--format") && (i + 1 < argc))
        {
            raw_format = argv[i + 1];
            if (raw_format != "dicom" && raw_format != "nifty")
            {
                std::cerr << "Error: raw data format must be 'dicom' or 'nifty'" << std::endl;
                return 1;
            }
            i++; // 跳过参数值
        }
        else if (arg == "--forward-project" && i + 2 < argc)
        {
            forward_image = argv[i + 1];
            forward_output = argv[i + 2];
            i += 2;
        }
        else if (arg == "--mu-map-cm" && i + 1 < argc)
        { mu_map_cm_path = argv[++i]; }
        else if (arg == "--export-mu-map-cm" && i + 1 < argc)
        { export_mu_map_cm_path = argv[++i]; }
        else if (arg == "--backproject" && i + 2 < argc)
        {
            backproject_proj = argv[i + 1];
            backproject_output = argv[i + 2];
            i += 2;
        }
        else if (arg == "--export-osem-measurement" && i + 2 < argc)
        {
            osem_main_output = argv[i + 1];
            osem_scatter_output = argv[i + 2];
            i += 2;
        }
        else if (arg == "--iterations" && i + 1 < argc)
        {
            iterations_override = std::stoi(argv[i + 1]);
            i++;
        }
        else if (arg == "--output" && i + 1 < argc)
        {
            output_override = argv[i + 1];
            i++;
        }
        else if (arg == "--disable-cdr")
        {
            cdr_disabled = true;
        }
    }

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    if (!mu_map_cm_path.empty() || !export_mu_map_cm_path.empty()) {
        const bool raw_forward = forward_image.size() >= 4 && forward_image.substr(forward_image.size()-4) == ".bin";
        if ((!raw_forward && backproject_proj.empty()) || !osem_main_output.empty()) {
            std::cerr << "mu-map options require raw .bin forward or backprojection.\n";
            return 2;
        }
    }
    Reconstructor reconstructor;
    reconstructor.set_mu_map_cm(mu_map_cm_path);
    reconstructor.set_export_mu_map_cm(export_mu_map_cm_path);
    if (!osem_main_output.empty())
    {
        reconstructor.parse(argv[1]);
        return reconstructor.export_osem_measurement_from_nifty(
            osem_main_output, osem_scatter_output);
    }
    if (!forward_image.empty())
    {
        reconstructor.parse(argv[1]);
        if (cdr_disabled)
            reconstructor.disable_cdr();
        if (forward_image.size() >= 4 && forward_image.substr(forward_image.size() - 4) == ".bin")
            return reconstructor.forward_project_from_rawdata(forward_image, forward_output);
        else
            return reconstructor.forward_project_from_nifty(forward_image, forward_output);
    }
    if (!backproject_proj.empty())
    {
        reconstructor.parse(argv[1]);
        if (cdr_disabled)
            reconstructor.disable_cdr();
        return reconstructor.backproject_from_rawdata(backproject_proj, backproject_output);
    }
    if (raw_format == "nifty")
    {
        reconstructor.parse(argv[1]);
        if (iterations_override > 0)
            reconstructor.set_num_iterations(iterations_override);
        if (!output_override.empty())
            reconstructor.set_output_path(output_override);
        if (cdr_disabled)
            reconstructor.disable_cdr();
        return reconstructor.run_from_nifty();
    }
    if (raw_format == "dicom")
    {
        reconstructor.parse_from_backend(argv[1]);
        reconstructor.run_from_backend();
    }
    else if (raw_format == "nifty")
    {
        reconstructor.parse(argv[1]);
        reconstructor.run_from_nifty();
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    // 计算时间差
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "Time taken by function: " << time_span.count() << " seconds." << std::endl;

    return 0;
}
