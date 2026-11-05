
#include <regex>
#include "io/dicom.h"

NMRECON_NAMESPACE_BEGIN
namespace io
{
    bool isValidUID(const char *uid)
    {
        // 验证：1. 只包含数字和点 2. 首字符为数字 3. 不超过64字符
        return std::regex_match(uid, std::regex("^[0-9]+(\\.[0-9]+)*$")) &&
               strlen(uid) <= 64;
    }

    void printTagValue(DcmElement *elem)
    {
        if (!elem)
            return;

        const DcmVR vr(elem->getVR());
        OFString value;

        switch (vr.getEVR())
        {
        // 字符串类VR
        case EVR_DA: // Date
        case EVR_TM: // Time
        case EVR_PN: // Person Name
        case EVR_CS: // Code String
        case EVR_LO: // Long String
        case EVR_SH: // Short String
        case EVR_UI: // UID
        case EVR_IS: // Integer String
        case EVR_DS: // Decimal String
        case EVR_ST: // Short Text
        {
            if (elem->getOFString(value, 0).good())
                std::cout << "Value: " << value << std::endl;
            else
                std::cout << "Value: <unreadable>" << std::endl;
            break;
        }

        // 整数类VR
        case EVR_US: // Unsigned Short (16-bit)
        {
            Uint16 val = 0;
            if (elem->getUint16(val, 0).good())
                std::cout << "Value: " << val << std::endl;
            else
                std::cout << "Value: <unreadable>" << std::endl;
            break;
        }
        case EVR_SS: // Signed Short (16-bit)
        {
            Sint16 val = 0;
            if (elem->getSint16(val, 0).good())
                std::cout << "Value: " << val << std::endl;
            else
                std::cout << "Value: <unreadable>" << std::endl;
            break;
        }
        case EVR_UL: // Unsigned Long (32-bit)
        {
            Uint32 val = 0;
            if (elem->getUint32(val, 0).good())
                std::cout << "Value: " << val << std::endl;
            else
                std::cout << "Value: <unreadable>" << std::endl;
            break;
        }
        case EVR_SL: // Signed Long (32-bit)
        {
            Sint32 val = 0;
            if (elem->getSint32(val, 0).good())
                std::cout << "Value: " << val << std::endl;
            else
                std::cout << "Value: <unreadable>" << std::endl;
            break;
        }

        // 浮点数类VR
        case EVR_FL: // Floating Point Single (32-bit)
        {
            Float32 val = 0.0f;
            if (elem->getFloat32(val, 0).good())
                std::cout << "Value: " << val << std::endl;
            else
                std::cout << "Value: <unreadable>" << std::endl;
            break;
        }
        case EVR_FD: // Floating Point Double (64-bit)
        {
            Float64 val = 0.0;
            if (elem->getFloat64(val, 0).good())
                std::cout << "Value: " << val << std::endl;
            else
                std::cout << "Value: <unreadable>" << std::endl;
            break;
        }

        // 二进制数据（像素数据等），这里简单打印长度
        case EVR_OB: // Other Byte String
        case EVR_OW: // Other Word String
        case EVR_OF: // Other Float String
        case EVR_UT: // Unlimited Text
        case EVR_UN: // Unknown
        {
            // 打印数据长度
            Uint32 length = elem->getLength();
            std::cout << "Value: [Binary data, length = " << length << "]" << std::endl;
            break;
        }

        default:
            std::cout << "Value: [Unsupported VR type: " << vr.getVRName() << "]" << std::endl;
            break;
        }
    }

    void copyTags(DcmDataset *src, DcmDataset *dest, const std::vector<DcmTagKey> &tags)
    {
        for (const auto &tag : tags)
        {
            DcmElement *elem = nullptr;
            if (src->findAndGetElement(tag, elem).good() && elem)
            {
                DcmElement *cloned = dynamic_cast<DcmElement *>(elem->clone());
                if (cloned)
                {
                    dest->insert(cloned, true);
                    std::cout << "Copied tag: " << tag.toString() << std::endl;
                    printTagValue(cloned);
                }
            }
        }
    }

    bool DicomProcessor::read_dicom_as_format(const std::string &filepath, DcmFileFormat &fileFormat)
    {
        // OFCondition status = fileFormat.loadFile(filepath);
        OFCondition status = fileFormat.loadFile(OFFilename(filepath.c_str()));
        if (status.bad())
        {
            std::cerr << "Error reading DICOM file: " << status.text() << std::endl;
            return false;
        }
        return true;
    }

    /**
     * 拷贝DICOM元信息,修改SOP instance UID
     * 用于重建过程中的元信息传递，将原始数据信息传递给重建图像
     */
    bool DicomProcessor::transfer_dicom_meta(DcmFileFormat &src, DcmFileFormat &dest,
                                             const char *newSOPInstanceUID)
    {

        if (!isValidUID(newSOPInstanceUID))
        {
            std::cerr << "Invalid UID format" << std::endl;
            return false;
        }
        DcmMetaInfo *srcMeta = src.getMetaInfo();
        DcmMetaInfo *destMeta = dest.getMetaInfo();
        if (!srcMeta || !destMeta)
        {
            std::cerr << "Error: Failed to get DICOM meta information" << std::endl;
            return false;
        }
        // can not explicitly set ImplementationClassUID;
        // std::string implementationUID;
        // OFCondition status = srcMeta->findAndGetOFString(DCM_ImplementationClassUID,implementationUID);
        // destMeta->putAndInsertString(DCM_ImplementationClassUID, implementationUID.c_str());

        std::vector<DcmTagKey> tagsToCopy = {
            DCM_PatientName, DCM_PatientID, DCM_PatientBirthDate,
            DCM_StudyInstanceUID, DCM_SeriesInstanceUID,
            DCM_StudyDate, DCM_StudyTime,
            DCM_Modality, DCM_InstitutionName,
            DCM_AccessionNumber, DCM_ReferringPhysicianName};
        DcmDataset *srcDataset = src.getDataset();
        DcmDataset *destDataset = dest.getDataset();
        copyTags(srcDataset, destDataset, tagsToCopy);

        destDataset->putAndInsertString(DCM_SOPInstanceUID, newSOPInstanceUID);
        destDataset->putAndInsertString(DCM_SOPClassUID, UID_NuclearMedicineImageStorage);
        return true;
    }

    /**
     * 加载SPECT原始投影数据,包括多个探头，多个能窗，多个周期
     */
    bool DicomProcessor::load_spect_raw(DcmDataset *dataset, std::vector<uint16_t> &data)
    {

        Uint16 rows = 0, cols = 0, frames = 0, samples = 1, bitsAllocated = 0;
        dataset->findAndGetUint16(DCM_Rows, rows);
        dataset->findAndGetUint16(DCM_Columns, cols);
        dataset->findAndGetUint16(DCM_SamplesPerPixel, samples);
        dataset->findAndGetUint16(DCM_BitsAllocated, bitsAllocated);
        // 判断是否有多帧
        OFString frameCountStr;
        if (dataset->findAndGetOFString(DCM_NumberOfFrames, frameCountStr).good())
        {
            frames = std::stoi(frameCountStr.c_str());
        }

        int width = cols;
        int height = rows;
        int depth = frames;

        // 检查位数
        if (bitsAllocated != 16)
        {
            std::cerr << "Only 16-bit DICOM images are supported in this function." << std::endl;
            return false;
        }
        // 读取像素数据
        const Uint16 *pixelData = nullptr;
        unsigned long pixelCount = 0;
        if (dataset->findAndGetUint16Array(DCM_PixelData, pixelData, &pixelCount).bad() || !pixelData)
        {
            std::cerr << "Failed to read DICOM pixel data." << std::endl;
            return false;
        }

        const size_t totalPixels = static_cast<size_t>(width) * height * depth;
        if (pixelCount < totalPixels)
        {
            std::cerr << "Pixel data size mismatch: expected " << totalPixels << " but got " << pixelCount << std::endl;
            return false;
        }
        std::cout << "pixelCount: " << pixelCount << ", data shape: " << width << "," << height << "," << depth << std::endl;
        data.assign(pixelData, pixelData + totalPixels);
        return true;
    }

    bool DicomProcessor::set_output_image(DcmDataset *destDataset,
                                          const std::vector<uint16_t> &image_vector,
                                          const nmrecon::common::Coordinate3D &image_meta)
    {
        auto image_size = image_meta.size;
        const size_t size[3] = {image_size.x, image_size.y, image_size.z};
        auto image_spacing = image_meta.spacing;
        const precision_t spacing[3] = {image_spacing.x, image_spacing.y, image_spacing.z};
        auto image_origin = image_meta.origin;
        const precision_t origin[3] = {image_origin.x, image_origin.y, image_origin.z};

        // 设置DICOM标签
        destDataset->putAndInsertString(DCM_SamplesPerPixel, "1");
        destDataset->putAndInsertString(DCM_PhotometricInterpretation, "MONOCHROME2");
        destDataset->putAndInsertString(DCM_Modality, "NM");
        destDataset->putAndInsertUint16(DCM_Rows, size[0]);
        destDataset->putAndInsertUint16(DCM_Columns, size[1]);
        destDataset->putAndInsertUint16(DCM_BitsAllocated, 16);
        destDataset->putAndInsertUint16(DCM_BitsStored, 16);
        destDataset->putAndInsertUint16(DCM_HighBit, 15);
        destDataset->putAndInsertUint16(DCM_PixelRepresentation, 0); // 0: unsigned

        char spacingStr[64], originStr[64];
        snprintf(spacingStr, sizeof(spacingStr), "%.4f\\%.4f", spacing[1], spacing[0]);
        snprintf(originStr, sizeof(originStr), "%.4f\\%.4f\\%.4f", origin[0], origin[1], origin[2]);
        

        destDataset->putAndInsertString(DCM_PixelSpacing, spacingStr);
        destDataset->putAndInsertString(DCM_ImageOrientationPatient, "1\\0\\0\\0\\1\\0");
        destDataset->putAndInsertString(DCM_ImagePositionPatient, originStr);
        destDataset->putAndInsertString(DCM_SliceThickness, std::to_string(spacing[2]).c_str());
        destDataset->putAndInsertString(DCM_SpacingBetweenSlices, std::to_string(-spacing[2]).c_str());
        destDataset->putAndInsertString(DCM_NumberOfFrames, std::to_string(size[2]).c_str());
        destDataset->putAndInsertString(DCM_NumberOfSlices, std::to_string(size[2]).c_str());

        const size_t totalPixels = static_cast<size_t>(size[0]) * size[1] * size[2];
        destDataset->putAndInsertUint16Array(DCM_PixelData, image_vector.data(), totalPixels);

        return true;
    }

    bool DicomProcessor::save_format_to_dicom(const std::string &filepath,
                                              DcmFileFormat &fileFormat,
                                              const E_TransferSyntax &xfer)
    {
        // OFCondition status = fileFormat.saveFile(filepath, xfer, EET_ExplicitLength, EGL_recalcGL, EPD_noChange, EWM_updateMeta);
          OFCondition status = fileFormat.saveFile(OFFilename(filepath.c_str()), xfer, EET_ExplicitLength, EGL_recalcGL, EPD_noChange, EWM_updateMeta);
        if (status.bad())
        {
            std::cerr << "Error saving DICOM file: " << status.text() << std::endl;
            return false;
        }
        return true;
    }

    bool make_scan_setting(DcmDataset *dataset)
    {
        OFString modality;
        if (dataset->findAndGetOFString(DCM_Modality, modality).bad() || modality != "NM")
        {
            std::cerr << "Not a valid SPECT projection DICOM file." << std::endl;
            return false;
        }

        // 读取探测器数量和旋转数量
        Uint16 num_detectors = 0, num_rotations = 0;
        bool hasDetectors = dataset->findAndGetUint16(DCM_NumberOfDetectors, num_detectors).good();
        bool hasRotations = dataset->findAndGetUint16(DCM_NumberOfRotations, num_rotations).good();
        if (!hasDetectors || !hasRotations)
        {
            std::cerr << "Failed to get NumberOfDetectors or NumberOfRotations from dataset." << std::endl;
            return false;
        }
        if (num_rotations > 1)
        {
            // 如果有多个旋转，则需要处理每个旋转的投影数据
            // 动态投影数据的处理逻辑
            std::vector<int> actual_frame_durations;
        }

        // 以下为静态投影数据的处理逻辑, num_rotations == 1
        std::vector<precision_t> detector_start_degree;
        std::vector<precision_t> detector_end_degree;
        std::vector<precision_t> rotation_radii;
        precision_t image_position_patient_on_z_axis;
        Float64 scan_arc = 0.0;
        Float64 angular_step = 0.0;
        Uint16 num_frames = 0;

        // 1. 读取旋转序列中的信息
        DcmSequenceOfItems *rotationSeq;
        if (dataset->findAndGetSequence(DCM_RotationInformationSequence, rotationSeq).bad())
        {
            std::cerr << "Failed to get RotationInformationSequence." << std::endl;
            return false;
        }
        for (Uint16 i = 0; i < num_rotations; ++i)
        { // 这里只处理一次旋转的情况
            DcmItem *item = rotationSeq->getItem(i);
            // 读取扫描弧度,角度步长和帧数
            if (item->findAndGetFloat64(DCM_ScanArc, scan_arc).bad() ||
                item->findAndGetFloat64(DCM_AngularStep, angular_step).bad() ||
                item->findAndGetUint16(DCM_NumberOfFramesInRotation, num_frames).bad())
            {
                std::cerr << "Failed to get ScanArc or AngularStep or NumberOfFramesInRotation from RotationInformationSequence." << std::endl;
                return false;
            }
            // 起始角度
            precision_t startAngle = 0.0;
            if (num_detectors == 1 && item->findAndGetFloat32(DCM_StartAngle, startAngle).good())
            {
                // 如果只有一个探测器，读取起始角度，否则不处理
                detector_start_degree.push_back(startAngle);
            }

            // 读取该次旋转的半径
            DcmElement *radialPosElem;
            if (item->findAndGetElement(DCM_RadialPosition, radialPosElem).good() && radialPosElem->getVR() == EVR_DS)
            {
                const size_t numValues = radialPosElem->getNumberOfValues();
                if (numValues == 1) // 如果只有一个值，表示所有探测器的半径相同
                {
                    Float64 val;
                    if (radialPosElem->getFloat64(val, 0).good())
                    {
                        // 插入相同的半径值到所有帧
                        rotation_radii.insert(rotation_radii.end(), num_frames * num_detectors, precision_t(val));
                    }
                    else
                    {
                        std::cerr << "Failed to parse RadialPosition value at index 0." << std::endl;
                        return false;
                    }
                }
            }
        }

        // 2. 读取探测器序列信息
        DcmSequenceOfItems *detectorSeq;
        if (dataset->findAndGetSequence(DCM_DetectorInformationSequence, detectorSeq).bad())
        {
            std::cerr << "Failed to get DetectorInformationSequence." << std::endl;
            return false;
        }
        for (Uint16 i = 0; i < num_detectors; ++i)
        {
            DcmItem *item = detectorSeq->getItem(i);

            // 读取image_position_patient_on_z_axis
            DcmElement *elem;
            if (item->findAndGetElement(DCM_ImagePositionPatient, elem).good() && elem->getVR() == EVR_DS)
            {
                const size_t numValues = elem->getNumberOfValues();
                if (numValues >= 3)
                {
                    Float64 val;
                    elem->getFloat64(val, 2); // 获取第一个像素的Z轴位置
                    image_position_patient_on_z_axis = precision_t(val);
                }
                else
                {
                    std::cerr << "ImagePositionPatient does not have enough values." << std::endl;
                    return false;
                }
            }

            // 读取探测器的旋转起始角度
            double startAngle = 0.0;
            if (num_detectors > 1 && item->findAndGetFloat64(DCM_StartAngle, startAngle).good())
            {
                // 如果有多个探测器，读取每个探测器的起始角度
                detector_start_degree.push_back(precision_t(startAngle));
            }

            // 读取探测器的旋转半径
            DcmElement *radialPosElem;
            if (item->findAndGetElement(DCM_RadialPosition, radialPosElem).good() && radialPosElem->getVR() == EVR_DS)
            {
                const size_t numValues = radialPosElem->getNumberOfValues();
                if (numValues > 1 && rotation_radii.empty())
                {
                    for (size_t j = 0; j < numValues; ++j)
                    {
                        Float64 val;
                        if (radialPosElem->getFloat64(val, j).good())
                        {
                            rotation_radii.push_back(precision_t(val));
                        }
                        else
                        {
                            std::cerr << "Failed to parse RadialPosition value at index " << j << std::endl;
                        }
                    }
                }
            }
        }
    }


    bool DicomProcessor::load_spect_projection(const std::string &filepath)
    {
        DcmFileFormat fileFormat;
        if (!read_dicom_as_format(filepath, fileFormat))
        {
            std::cerr << "Failed to read DICOM file: " << filepath << std::endl;
            return false;
        }
        DcmDataset *dataset = fileFormat.getDataset();
        // 检查是否为SPECT投影数据
    }
};
NMRECON_NAMESPACE_END