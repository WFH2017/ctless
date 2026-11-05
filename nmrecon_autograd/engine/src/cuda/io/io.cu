
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <cuda_runtime.h>

#include "common/basic_type.h"
#include "io/itk_warpper.h"
#include "io/common.h"

#include "cuda/io/utils.h"
#include "cuda/io/io.h"
#include "cuda/reconstruction/scatter.h"

NMRECON_NAMESPACE_BEGIN
namespace io
{

    void load_from_dicom(DcmFileFormat &raw_dcm,
                         ProjectionValue &proj_value,
                         ProjectionValue &scatter_value)
    {
        std::vector<uint16_t> projection_dicom;
        DicomProcessor::load_spect_raw(raw_dcm.getDataset(), projection_dicom);

        // 使用封装的 pinned buffer 创建逻辑
        HostPinnedBuffer pinned_buffer = convert_to_pinned_buffer(projection_dicom);

        // 调用通用拷贝函数
        copy_from_pinned_to_gpu(pinned_buffer, proj_value, scatter_value);
    }

    void load_from_nifty(const ScanInfo &scan_info,
                         ProjectionValue &proj_value,
                         ProjectionValue &scatter_value)
    {
        ;
        auto projection_nifty = load_spect_nifty_cpu(scan_info.rawdata_path);
        HostPinnedBuffer pinned_buffer = convert_to_pinned_buffer(projection_nifty);

        // Step 2: 使用通用 GPU 拷贝函数
        copy_from_pinned_to_gpu(pinned_buffer, proj_value, scatter_value);
    }

    nmrecon::common::DeviceVector load_spect_nifty_image(const std::string &filepath)
    {
        auto input = load_spect_nifty_cpu(filepath);
        thrust::host_vector<float> host(input.size());
        for (size_t index = 0; index < input.size(); ++index)
            host[index] = static_cast<float>(input[index]);
        nmrecon::common::DeviceVector device(host.size());
        thrust::copy(host.begin(), host.end(), device.begin());
        return device;
    }

    // 从dicom中读取投影数据
    std::vector<nmrecon::reconstruction::rawdata::Projection> load_projection(const nmrecon::io::ScanInfo &scan_info,
                                                                              const std::string &scatter_type,
                                                                              DcmFileFormat &raw_dcm)
    {
        auto scanner = nmrecon::scanner::SPECT(scan_info.scanner.setting);

        size_t total_num_rays = scanner.get_total_num_rays();

        std::cout << "Total number of rays: " << total_num_rays << std::endl;
        std::cout << "start to load raw data.\n";

        ProjectionValue proj_value(total_num_rays, 0);
        ProjectionValue scatter_value(total_num_rays, 0);

        auto proj_path = scan_info.rawdata_path;
        // load rawdata
        if (DicomProcessor::read_dicom_as_format(proj_path, raw_dcm))
        {
            load_from_dicom(raw_dcm, proj_value, scatter_value);
        }

        // std::cout << "debug proj_value size: " << proj_value.size() << std::endl;
        nmrecon::reconstruction::rawdata::Projection projection(scanner.get_trajectory_list(),
                                                                scanner.get_collimator().get_coordinate_prototype().size,
                                                                proj_value);

        nmrecon::reconstruction::rawdata::Projection scatter_raw(scanner.get_trajectory_list(),
                                                                 scanner.get_collimator().get_coordinate_prototype().size,
                                                                 scatter_value);

        // butterworth_filter cutoff freq, order
        nmrecon::reconstruction::correction::scatter::smooth_projection(projection, 0.3, 5);
        nmrecon::reconstruction::correction::scatter::smooth_projection(scatter_raw, 0.3, 5);

        // TODO:需要先考虑总的投影中加上散射分量
        if (scatter_type == "NSC")
        {
            // 如果是非散射校正，则将散射投影置为0.
            scatter_raw.fill(0.0f);
        }

        std::vector<nmrecon::reconstruction::rawdata::Projection> projections = {projection, scatter_raw * 0.55};

        return projections;
    }
    // 从nifty中读取投影数据
    // 从 NIfTI 中读取投影数据
    std::vector<nmrecon::reconstruction::rawdata::Projection> load_projection(const nmrecon::io::ScanInfo &scan_info,
                                                                              const std::string &scatter_type)
    {
        // 构造 scanner 对象
        auto scanner = nmrecon::scanner::SPECT(scan_info.scanner.setting);

        size_t total_num_rays = scanner.get_total_num_rays();

        std::cout << "Total number of rays: " << total_num_rays << std::endl;
        std::cout << "Start to load raw data from NIfTI.\n";

        // 申请投影数据容器
        ProjectionValue proj_value(total_num_rays, 0);
        ProjectionValue scatter_value(total_num_rays, 0);
        // 从 NIfTI 文件加载数据到 proj_value
        load_from_nifty(scan_info, proj_value, scatter_value);

        // 根据proj_value创建投影对象
        nmrecon::reconstruction::rawdata::Projection projection(scanner.get_trajectory_list(),
                                                                scanner.get_collimator().get_coordinate_prototype().size,
                                                                proj_value);

        nmrecon::reconstruction::rawdata::Projection scatter_raw(scanner.get_trajectory_list(),
                                                                 scanner.get_collimator().get_coordinate_prototype().size,
                                                                 scatter_value);

        // 对投影和散射数据做平滑滤波处理
        nmrecon::reconstruction::correction::scatter::smooth_projection(projection, 0.3, 5);
        nmrecon::reconstruction::correction::scatter::smooth_projection(scatter_raw, 0.3, 5);

        if (scatter_type == "NSC")
        {
            // 非散射校正，将散射投影置0
            scatter_raw.fill(0.0f);
        }

        // 返回两个投影，散射投影乘以0.55权重
        return {projection, scatter_raw * 0.55f};
    }

}
NMRECON_NAMESPACE_END
