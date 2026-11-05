
#include <itkGDCMImageIO.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkNiftiImageIO.h>

#include <itkImageFileReader.h>
#include <itkImageSeriesReader.h>
#include <itkResampleImageFilter.h>
#include <itkIdentityTransform.h>                       // 通常用于定义输出网格，如果输出网格已经通过参数直接指定
#include <itkLinearInterpolateImageFunction.h>          // 常用插值器
#include <itkNearestNeighborInterpolateImageFunction.h> // 另一种插值器
#include <itkBSplineInterpolateImageFunction.h>         // 更高级的插值器
#include <itkThresholdImageFilter.h>

#include <itkImageFileWriter.h>     // for saving images
#include <itkMetaImageIOFactory.h>  // for .mha format
#include <itkNiftiImageIOFactory.h> // for .nii format
#include <itkGDCMImageIOFactory.h>  // for .dcm (DICOM) format

#include "io/itk.h"

NMRECON_NAMESPACE_BEGIN
namespace io
{

    void ITKProcessor::print_ct_image_info(ItkCTImageType::Pointer &image, const std::string &prefix)
    {
        if (!image)
        {
            std::cerr << "[WARNING] Image pointer is null. Cannot print info." << std::endl;
            return;
        }

        const auto &origin = image->GetOrigin();
        const auto &spacing = image->GetSpacing();
        const auto &size = image->GetLargestPossibleRegion().GetSize();
        const auto &direction = image->GetDirection();

        std::cout << prefix << std::endl;
        std::cout << "  Origin   : " << origin << std::endl;
        std::cout << "  Spacing  : " << spacing << std::endl;
        std::cout << "  Size     : " << size << std::endl;
        std::cout << "  Direction:" << std::endl;
        for (unsigned int i = 0; i < 3; ++i)
        {
            std::cout << "    ";
            for (unsigned int j = 0; j < 3; ++j)
            {
                std::cout << std::fixed << std::setprecision(6) << direction(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    bool ITKProcessor::replace_ct_padding_with_air(ItkCTImageType::Pointer &image, CtPixelType airValue)
    {
        using ThresholdFilterType = itk::ThresholdImageFilter<ItkCTImageType>;
        auto thresholdFilter = ThresholdFilterType::New();

        thresholdFilter->SetInput(image);
        thresholdFilter->ThresholdBelow(airValue);
        thresholdFilter->SetOutsideValue(airValue);

        try
        {
            thresholdFilter->Update();
            image = thresholdFilter->GetOutput();

            std::cout << "[INFO] CT padding values below " << airValue
                      << " replaced with air value " << airValue << std::endl;

            return true;
        }
        catch (itk::ExceptionObject &e)
        {
            std::cerr << "[ERROR] Failed to replace padding in CT image: " << e << std::endl;
            return false;
        }
    }

    bool ITKProcessor::load_ct_dicom(const std::string &file_dir, ItkCTImageType::Pointer &image)
    {
        // 创建DICOM序列文件名生成器
        auto seriesNames = itk::GDCMSeriesFileNames::New();
        seriesNames->SetDirectory(file_dir);

        const auto &seriesUIDs = seriesNames->GetSeriesUIDs();
        if (seriesUIDs.empty())
        {
            std::cerr << "Error: No DICOM series found in directory." << std::endl;
            return false;
        }

        std::string ctSeriesUID = "";
        std::vector<std::string> ctFilenames;

        // 遍历所有找到的系列UID

        for (const auto &uid : seriesUIDs)
        {
            const auto &filenamesInSeries = seriesNames->GetFileNames(uid);
            if (filenamesInSeries.empty())
            {
                continue; // 跳过空的系列
            }

            // 读取该系列中的第一个文件以获取Modality信息
            // 创建一个临时的ImageFileReader，并设置其ImageIO和文件名
            auto tempFileReader = itk::ImageFileReader<ItkCTImageType>::New(); // Changed from firstFileReader
            auto tempGdcmIO = itk::GDCMImageIO::New();                         // Changed from gdcmIOForModality
            tempFileReader->SetImageIO(tempGdcmIO);
            tempFileReader->SetFileName(filenamesInSeries[0]);

            try
            {
                // 注意：这里只读取头信息，不读取像素数据，所以是轻量级的
                tempFileReader->UpdateOutputInformation();
                itk::MetaDataDictionary &dict = tempGdcmIO->GetMetaDataDictionary();
                std::string modalityValue;
                itk::ExposeMetaData<std::string>(dict, "0008|0060", modalityValue); // DICOM Tag for Modality

                if (modalityValue == "CT") // 检查Modality是否为"CT"
                {
                    ctSeriesUID = uid;
                    ctFilenames = filenamesInSeries;
                    break; // 找到CT系列后立即跳出循环
                }
            }
            catch (itk::ExceptionObject &e)
            {
                std::cerr << "Warning: Could not read DICOM information from file " << filenamesInSeries[0] << ": " << e << std::endl;
                // 继续尝试下一个系列
            }
        }

        if (ctSeriesUID.empty())
        {
            std::cerr << "Error: No CT DICOM series found in directory." << std::endl;
            return false;
        }

        // 使用找到的CT系列的文件名来读取图像
        auto reader = itk::ImageSeriesReader<ItkCTImageType>::New();
        auto gdcmIO = itk::GDCMImageIO::New(); // 为主读取器创建新的IO对象
        reader->SetImageIO(gdcmIO);
        reader->SetFileNames(ctFilenames);
        try
        {
            reader->Update();
            image = reader->GetOutput();

            if (!replace_ct_padding_with_air(image, -1000))
            {
                std::cerr << "[WARNING] Failed to process padding values in CT image." << std::endl;
            }

            // 获取元数据并打印（保持不变）
            std::string prefix = "Successfully loaded CT series with UID: " + ctSeriesUID;
            print_ct_image_info(image, prefix);

            return true;
        }
        catch (itk::ExceptionObject &e)
        {
            std::cerr << "Error loading CT series: " << e << std::endl;
            return false;
        }
    }

    bool ITKProcessor::load_ct_nifty(const std::string &filepath, ItkCTImageType::Pointer &outputImage)
    {
        try
        {
            using ReaderType = itk::ImageFileReader<ItkCTImageType>;
            ReaderType::Pointer reader = ReaderType::New();

            reader->SetFileName(filepath);

            // 设置为 NIfTI 读取器，虽然 ITK 通常能自动识别，但更安全
            itk::NiftiImageIO::Pointer niftiIO = itk::NiftiImageIO::New();
            reader->SetImageIO(niftiIO);

            reader->Update();
            outputImage = reader->GetOutput();
            // 替换 padding 区域为标准空气值（-1000 HU）
            if (!replace_ct_padding_with_air(outputImage, -1000))
            {
                std::cerr << "[WARNING] Failed to process padding values in NIfTI CT image." << std::endl;
            }
            std::string prefix = "Successfully loaded CT";
            print_ct_image_info(outputImage, prefix);
            return true;
        }
        catch (itk::ExceptionObject &ex)
        {
            std::cerr << "[ERROR] Failed to load NIfTI image: " << filepath << "\n"
                      << ex << std::endl;
            return false;
        }
    }
    bool ITKProcessor::load_spect_nifty(const std::string &filepath, std::vector<uint16_t> &data)
    {
        using ReaderType = itk::ImageFileReader<ItkNmImageType>;
        ReaderType::Pointer reader = ReaderType::New();
        reader->SetFileName(filepath);

        try
        {
            reader->Update();
        }
        catch (const itk::ExceptionObject &e)
        {
            throw std::runtime_error("ITK image reading failed: " + std::string(e.GetDescription()));
        }

        ItkNmImageType::Pointer image = reader->GetOutput();
        if (!image)
        {
            throw std::runtime_error("Failed to get ITK image output.");
        }

        const uint16_t *buffer = image->GetBufferPointer();
        if (!buffer)
        {
            throw std::runtime_error("Image buffer pointer is null.");
        }

        const size_t num_elements = image->GetLargestPossibleRegion().GetNumberOfPixels();

        data.resize(num_elements);
        std::copy(buffer, buffer + num_elements, data.begin());

        return true;
    }

    template <typename TImageType>
    bool ITKProcessor::save_itk_image(const std::string &file_path, typename TImageType::Pointer &image)
    {
        if (image.IsNull())
        {
            std::cerr << "Error: Cannot save a null ITK image pointer." << std::endl;
            return false;
        }

        // 注册所有常用的ImageIO工厂，以便ITK自动识别文件类型
        // 建议在程序启动时只执行一次这些注册，而不是每次保存都注册
        // 这里为了函数独立性暂时放在这里，但更优的做法是在main函数或初始化函数中调用：
        // itk::MetaImageIOFactory::RegisterOneFactory();
        // itk::NiftiImageIOFactory::RegisterOneFactory();
        // itk::GDCMImageIOFactory::RegisterOneFactory(); // for DICOM series output

        // 创建ImageFileWriter

        auto writer = itk::ImageFileWriter<TImageType>::New();
        writer->SetFileName(file_path);
        writer->SetInput(image);

        try
        {
            writer->Update();
            std::cout << "Image successfully saved to: " << file_path << std::endl;
            return true;
        }
        catch (itk::ExceptionObject &e)
        {
            std::cerr << "Error saving image to " << file_path << ": " << e << std::endl;
            return false;
        }
    }

    /**
     * 将一个std::vector转换为ITK图像
     */
    template <typename TImageType, typename TImagePixelType>
    typename TImageType::Pointer ITKProcessor::convert_vector_to_itk_image(
        const std::vector<TImagePixelType> &pixel_data_vector,
        const SpectFovParameters &fov_para)
    {
        // === 修正点：从 fov_para 中获取几何参数 ===
        const auto &image_size = fov_para.Size;           // 直接访问 SpectFovParameters 的 Size 成员
        const auto &image_spacing = fov_para.Spacing;     // 直接访问 SpectFovParameters 的 Spacing 成员
        const auto &image_origin = fov_para.Origin;       // 直接访问 SpectFovParameters 的 Origin 成员
        const auto &image_direction = fov_para.Direction; // 直接访问 SpectFovParameters 的 Direction 成员

        // 1. 验证输入数据尺寸
        unsigned long expected_num_pixels = 1;
        for (unsigned int i = 0; i < Dimension; ++i)
        {
            expected_num_pixels *= image_size[i];
        }

        if (pixel_data_vector.size() != expected_num_pixels)
        {
            std::cerr << "错误：像素数据 vector 的大小 (" << pixel_data_vector.size()
                      << ") 与预期图像尺寸 (" << expected_num_pixels << ") 不匹配。" << std::endl;
            return nullptr;
        }

        typename TImageType::Pointer image = TImageType::New(); // 使用模板类型
        typename TImageType::RegionType region;

        region.SetSize(image_size); // 设置图像的尺寸 (X, Y, Z)

        // 4. 为 ITK 图像分配内存
        image->SetRegions(region);
        image->Allocate();

        // 5. 从 std::vector 复制数据到 ITK 图像的缓冲区
        // ITK 的内部缓冲区是连续的，类似于一个扁平的 C 风格数组。
        // 我们假设 std::vector 中的数据顺序与 ITK 内部内存顺序一致。
        // 对于 3D 图像，ITK 的默认内存顺序通常是 Z-Y-X (即切片-行-列的顺序)。
        // 如果您的 std::vector 数据不是这个顺序，您需要在这里进行转换/重新排序。
        // 使用 memcpy 进行高效的内存拷贝
        memcpy(image->GetBufferPointer(), pixel_data_vector.data(), pixel_data_vector.size() * sizeof(TImagePixelType));

        // 6. 设置图像元数据 (空间信息，至关重要)
        image->SetSpacing(image_spacing);     // 设置体素的物理间距
        image->SetOrigin(image_origin);       // 设置图像原点的物理坐标
        image->SetDirection(image_direction); // 设置方向余弦矩阵

        // 7. 将创建的图像赋值给输出指针

        return image;
    }

    // 函数模板，可以处理任何 ITK 图像类型
    template <typename TImageType>
    std::vector<typename TImageType::PixelType>
    ITKProcessor::convert_itk_image_to_vector(typename TImageType::Pointer input_image)
    {
        // 1. 检查输入图像是否有效
        if (input_image.IsNull())
        {
            std::cerr << "错误：输入的 ITK 图像指针为空，无法提取数据。" << std::endl;
            return std::vector<typename TImageType::PixelType>(); // 返回空 vector
        }

        // 2. 获取图像的总像素数量
        const unsigned long num_pixels = input_image->GetLargestPossibleRegion().GetNumberOfPixels();

        if (num_pixels == 0)
        {
            std::cerr << "警告：ITK 图像不包含像素，将返回空 vector。" << std::endl;
            return std::vector<typename TImageType::PixelType>(); // 返回空 vector
        }

        // 3. 获取指向 ITK 图像内部缓冲区的指针
        // GetBufferPointer() 返回的是可读写的指针。如果图像是 const，需要 const 版本。
        // 这里假设图像是可读的，所以用 const
        const typename TImageType::PixelType *itk_buffer_ptr = input_image->GetBufferPointer();

        // 4. 使用指针和元素数量直接构造 std::vector
        // 这是最简洁高效的方式，因为它通常会直接从指针指向的内存区域拷贝数据
        std::vector<typename TImageType::PixelType> output_vector(
            itk_buffer_ptr,
            itk_buffer_ptr + num_pixels);

        std::cout << "成功将 ITK 图像数据提取到 std::vector 中。像素数量: " << output_vector.size() << std::endl;
        return output_vector;
    }

    SpectFovParameters ITKProcessor::compute_fov_parameters(const nmrecon::common::Coordinate3D &image_meta)
    {
        SpectFovParameters fov_params;
        // 填充 Origin (原点)
        fov_params.Origin[0] = image_meta.origin.x;
        fov_params.Origin[1] = image_meta.origin.y;
        fov_params.Origin[2] = image_meta.origin.z;

        // 填充 Spacing (间距)
        fov_params.Spacing[0] = image_meta.spacing.x;
        fov_params.Spacing[1] = image_meta.spacing.y;
        fov_params.Spacing[2] = image_meta.spacing.z;

        // 填充 Size (尺寸)
        fov_params.Size[0] = image_meta.size.x;
        fov_params.Size[1] = image_meta.size.y;
        fov_params.Size[2] = image_meta.size.z;

        // 填充 Direction (方向矩阵)
        // 假设 image_meta.direction_matrix 是一个 3x3 的 double 数组，
        // 且其索引顺序与 itk::Matrix 的 (row, col) 索引一致
        fov_params.Direction(0, 0) = image_meta.direction.m11;
        fov_params.Direction(0, 1) = image_meta.direction.m12;
        fov_params.Direction(0, 2) = image_meta.direction.m13;
        fov_params.Direction(1, 0) = image_meta.direction.m21;
        fov_params.Direction(1, 1) = image_meta.direction.m22;
        fov_params.Direction(1, 2) = image_meta.direction.m23;
        fov_params.Direction(2, 0) = image_meta.direction.m31;
        fov_params.Direction(2, 1) = image_meta.direction.m32;
        fov_params.Direction(2, 2) = image_meta.direction.m33;

        // 打印信息进行验证 (可选)
        std::cout << "Created FOV Parameters:" << std::endl;
        std::cout << "  Origin: " << fov_params.Origin << std::endl;
        std::cout << "  Spacing: " << fov_params.Spacing << std::endl;
        std::cout << "  Size: " << fov_params.Size << std::endl;
        std::cout << "  Direction:\n"
                  << fov_params.Direction << std::endl;

        return fov_params; // 返回填充好的结构体
    }

    template <typename TImageType, typename TImagePixelType>
    bool ITKProcessor::resample_image_to_fov(
        typename TImageType::Pointer input_image,
        const SpectFovParameters &output_fov_params,
        typename TImageType::Pointer &output_resampled_image,
        TImagePixelType default_pixel_value)
    {
        if (input_image.IsNull())
        {
            std::cerr << "错误：输入的 CT 图像指针为空，无法进行重采样。" << std::endl;
            return false;
        }

        // 定义重采样滤波器类型
        using ResampleFilterType = itk::ResampleImageFilter<TImageType, TImageType>;
        auto resampleFilter = ResampleFilterType::New();

        // 1. 设置输入图像
        resampleFilter->SetInput(input_image);

        // 2. 设置变换 (Transform)
        // 这里我们不是将输入图像进行变换，而是定义输出图像的网格。
        // 所以使用 IdentityTransform，因为我们直接设置输出图像的Origin, Spacing, Direction。
        using TransformType = itk::IdentityTransform<double, Dimension>;
        auto transform = TransformType::New();
        transform->SetIdentity();
        resampleFilter->SetTransform(transform);

        // 3. 设置插值器 (Interpolator)
        // 对于CT图像，线性插值通常是一个好的选择，因为它能平滑过渡。
        // 如果需要更高阶的插值，可以使用 itk::BSplineInterpolateImageFunction。
        // 如果需要最快的（但质量最低的）插值，可以使用 itk::NearestNeighborInterpolateImageFunction。
        using InterpolatorType = itk::LinearInterpolateImageFunction<TImageType, double>;
        auto interpolator = InterpolatorType::New();
        resampleFilter->SetInterpolator(interpolator);

        // 4. 设置默认像素值 (Default Pixel Value)
        // 当输出图像的某些区域在输入图像中没有对应像素时，会填充此值。
        // 对于CT图像，设置为空气的HU值 (-1000) 通常是合适的。
        resampleFilter->SetDefaultPixelValue(default_pixel_value);

        // 5. 设置输出图像的几何参数 (最关键的部分，定义SPECT FOV)
        // 这些参数直接定义了重采样后输出图像的网格和物理空间。
        resampleFilter->SetOutputOrigin(output_fov_params.Origin);
        resampleFilter->SetOutputSpacing(output_fov_params.Spacing);
        resampleFilter->SetOutputDirection(output_fov_params.Direction);
        resampleFilter->SetSize(output_fov_params.Size); // 输出图像的像素尺寸

        // 6. 执行重采样
        try
        {
            resampleFilter->Update();
            output_resampled_image = resampleFilter->GetOutput();
            std::cout << "图像成功重采样到目标FOV。" << std::endl;

            // 打印输出图像的元数据进行验证
            std::cout << "重采样后 图像信息:" << std::endl;
            std::cout << "  Origin: " << output_resampled_image->GetOrigin() << std::endl;
            std::cout << "  Spacing: " << output_resampled_image->GetSpacing() << std::endl;
            std::cout << "  Size: " << output_resampled_image->GetLargestPossibleRegion().GetSize() << std::endl;
            std::cout << "  Direction:\n"
                      << output_resampled_image->GetDirection() << std::endl;

            return true;
        }
        catch (itk::ExceptionObject &e)
        {
            std::cerr << "重采样 图像时发生错误: " << e << std::endl;
            return false;
        }
    }

    template bool ITKProcessor::save_itk_image<ItkCTImageType>(
        const std::string &,
        ItkCTImageType::Pointer &);

    template bool ITKProcessor::save_itk_image<ItkNmImageType>(
        const std::string &,
        ItkNmImageType::Pointer &);

    // Explicit instantiation for resample_image_to_fov
    template bool ITKProcessor::resample_image_to_fov<ItkCTImageType>(
        ItkCTImageType::Pointer,
        const SpectFovParameters &,
        ItkCTImageType::Pointer &,
        ItkCTImageType::PixelType);

    template bool ITKProcessor::resample_image_to_fov<ItkNmImageType>(
        ItkNmImageType::Pointer,
        const SpectFovParameters &,
        ItkNmImageType::Pointer &,
        ItkNmImageType::PixelType);

    template ItkNmImageType::Pointer ITKProcessor::convert_vector_to_itk_image<ItkNmImageType>(
        const std::vector<ItkNmImageType::PixelType> &,
        const SpectFovParameters &);

    template std::vector<nmrecon::io::CtPixelType> ITKProcessor::convert_itk_image_to_vector<nmrecon::io::ItkCTImageType>(
        typename nmrecon::io::ItkCTImageType::Pointer);

}
NMRECON_NAMESPACE_END