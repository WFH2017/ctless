#pragma once
#include <cstdint>
#include <itkImage.h>
#include "common/namespace.h"
#include "common/basic_type.h"
#include "io/common.h"

NMRECON_NAMESPACE_BEGIN

namespace io
{
    const unsigned int Dimension = 3;

    typedef std::uint16_t NmPixelType;
    typedef std::int16_t CtPixelType;

    typedef itk::Image<CtPixelType, Dimension> ItkCTImageType;
    typedef itk::Image<NmPixelType, Dimension> ItkNmImageType;

    // --- SPECT FOV 参数结构体 (不需要模板化，因为它是描述FOV的，不直接是图像类型) ---
    // 但是，其中的成员类型应该与所操作的图像维度匹配，这里使用ItkCTImageType的类型定义，
    // 因为物理参数通常是双精度浮点数，且维度是一致的。
    struct SpectFovParameters
    {
        ItkNmImageType::PointType Origin;        // SPECT FOV 的物理原点
        ItkNmImageType::SpacingType Spacing;     // SPECT FOV 的体素间距
        ItkNmImageType::SizeType Size;           // SPECT FOV 的像素尺寸
        ItkNmImageType::DirectionType Direction; // SPECT FOV 的方向余弦矩阵
    };



    struct ITKProcessor
    {

        static SpectFovParameters compute_fov_parameters(const nmrecon::common::Coordinate3D &image_meta);

        static void print_ct_image_info(ItkCTImageType::Pointer &image, const std::string &prefix);

        static bool replace_ct_padding_with_air(ItkCTImageType::Pointer &image, CtPixelType airValue);

        static bool load_ct_dicom(const std::string &filepath, ItkCTImageType::Pointer &outputImage);

        static bool load_ct_nifty(const std::string &filepath, ItkCTImageType::Pointer &outputImage);

        static bool load_spect_nifty(const std::string &filepath, std::vector<uint16_t> &outputImage);


        template <typename TImageType>
        static bool save_itk_image(const std::string &filepath, typename TImageType::Pointer &image);

        // --- 模板化 convert_vector_to_itk_image 函数 ---
        // TImageType 可以是 ItkCTImageType 或 ItkNmImageType
        // TImagePixelType 是从 TImageType 中提取的像素类型
        template <typename TImageType, typename TImagePixelType = typename TImageType::PixelType>
        static typename TImageType::Pointer convert_vector_to_itk_image(
            const std::vector<TImagePixelType> &pixel_data_vector,
            const SpectFovParameters &image_meta);

        template <typename TImageType>
        static std::vector<typename TImageType::PixelType> convert_itk_image_to_vector(
            typename TImageType::Pointer input_image);

        // --- 模板化 resample_ct_for_spect 函数 ---
        // InputImageType 可以是 ItkCTImageType
        // OutputImageType 可以是 ItkCTImageType (因为是重采样CT本身)
        // InputImagePixelType 是从 InputImageType 中提取的像素类型
        template <typename InputImageType, typename InputImagePixelType = typename InputImageType::PixelType>
        static bool resample_image_to_fov( // 重命名为更通用的 resample_image_to_fov
            typename InputImageType::Pointer input_image,
            const SpectFovParameters &output_fov_params, // 目标FOV
            typename InputImageType::Pointer &output_resampled_image,
            InputImagePixelType default_pixel_value); // 默认像素值现在是模板类型
    };

}

NMRECON_NAMESPACE_END