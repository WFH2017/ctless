#include "io/scan_info.h"
#ifndef NO_DCMTK
// 添加在 NMRECON_NAMESPACE_BEGIN 之前
inline bool getDicomString(DcmItem* item, const DcmTagKey& tag, std::string& output) {
    OFString ofStr;
    if (item->findAndGetOFString(tag, ofStr).good()) {
        output = ofStr.c_str();
        return true;
    }
    return false;
}
#endif

NMRECON_NAMESPACE_BEGIN
namespace io
{

    // 从 JSON 解析 ScanSetting
    void ScanSetting::from_json(const json &j)
    {
        collimator_shape = j.at("collimator_shape").get<std::string>();
        collimator_length = j.at("collimator_length").get<precision_t>();
        num_views = j.at("num_views").get<int>();
        num_detectors = j.at("num_detectors").get<int>();
        num_rotations = j.at("num_rotations").get<int>();

        // --- 对 rotation_radii 的判断和处理 ---
        // 先获取 rotation_radii 对应的 json 对象
        const json &rotation_radii_json = j.at("rotation_radii");
        if (!rotation_radii_json.is_array())
        {
            throw std::runtime_error("'rotation_radii' must be an array.");
        }

        std::vector<precision_t> parsed_radii = rotation_radii_json.get<std::vector<precision_t>>();
        if (parsed_radii.size() == 1)
        {
            // 扩展为全部视角的长度
            precision_t single_radius = parsed_radii[0];
            rotation_radii.assign(num_views * num_detectors, single_radius);
            std::cerr << "[INFO]: 'rotation_radii' has one value (" << single_radius
                      << "). Expanded to vector of " << num_views * num_detectors << " elements.\n";
        }
        else if (parsed_radii.size() == static_cast<size_t>(num_views * num_detectors))
        {
            rotation_radii = std::move(parsed_radii);
        }
        else
        {
            std::cerr << "[WARNING]: 'rotation_radii' array size (" << parsed_radii.size()
                      << ") does not match expected size (" << num_views * num_detectors << ").\n";
            throw std::runtime_error("rotation_radii size mismatch.");
        }


        detector_start_degree = j.at("detector_start_degree").get<std::vector<precision_t>>();
        detector_end_degree = j.at("detector_end_degree").get<std::vector<precision_t>>();
        detector_coordinate.from_json(j.at("detector_coordinate"));
        image_orientation_patient_on_z_axis = j.at("image_orientation_patient_on_z_axis").get<precision_t>();
        image_position_patient_on_z_axis = j.at("image_position_patient_on_z_axis").get<precision_t>();
        table_height = j.at("table_height").get<precision_t>();
        table_traverse = j.at("table_traverse").get<precision_t>();
    }
    json ScanSetting::to_json() const
    {
        return json{
            {"collimator_shape", collimator_shape},
            {"collimator_length", collimator_length},
            {"num_views", num_views},
            {"rotation_radii", rotation_radii},
            {"num_detectors", num_detectors},
            {"num_rotations", num_rotations},
            {"detector_start_degree", detector_start_degree},
            {"detector_end_degree", detector_end_degree},
            {"detector_coordinate", detector_coordinate.to_json()},
            {"image_position_patient_on_z_axis", image_position_patient_on_z_axis},
            {"image_orientation_patient_on_z_axis", image_orientation_patient_on_z_axis},
            {"table_height", table_height},
            {"table_traverse", table_traverse}};
    }

#ifndef NO_DCMTK
    void ScanSetting::from_dicom(DcmDataset *dataset)
    {
        // 暂时不需要这个参数
        collimator_length = 10.0; // 默认值

        OFString modality;
        if (dataset->findAndGetOFString(DCM_Modality, modality).bad() || modality != "NM")
        {
            std::cerr << "Not a valid SPECT projection DICOM file." << std::endl;
        }

        // 读取探测器数量和旋转数量
        num_rotations = 1;
        bool hasDetectors = dataset->findAndGetUint16(DCM_NumberOfDetectors, num_detectors).good();
        bool hasRotations = dataset->findAndGetUint16(DCM_NumberOfRotations, num_rotations).good();
        if (!hasDetectors || !hasRotations)
        {
            std::cerr << "Failed to get NumberOfDetectors or NumberOfRotations from dataset." << std::endl;
        }
        if (num_rotations > 1)
        {
            // 如果有多个旋转，则需要处理每个旋转的投影数据
            // 动态投影数据的处理逻辑
            std::vector<int> actual_frame_durations;
        }

        Float64 scan_arc = 0.0;
        Float64 angular_step = 0.0;
        OFString rotation_direction;
        Float64 table_height_value = 0.0;
        Float64 table_traverse_value = 0.0;

        // 1. 读取旋转序列中的信息
        DcmSequenceOfItems *rotationSeq;
        if (dataset->findAndGetSequence(DCM_RotationInformationSequence, rotationSeq).bad())
        {
            std::cerr << "Failed to get RotationInformationSequence." << std::endl;
        }
        for (Uint16 i = 0; i < num_rotations; ++i)
        { // 这里只处理一次旋转的情况
            DcmItem *item = rotationSeq->getItem(i);
            // 读取扫描弧度,角度步长和帧数
            if (item->findAndGetFloat64(DCM_ScanArc, scan_arc).bad() ||
                item->findAndGetFloat64(DCM_AngularStep, angular_step).bad() ||
                item->findAndGetUint16(DCM_NumberOfFramesInRotation, num_views).bad() ||
                item->findAndGetFloat64(DCM_TableHeight, table_height_value).bad() ||
                item->findAndGetFloat64(DCM_TableTraverse, table_traverse_value).bad())
            {
                std::cerr << "Failed to get [ScanArc, AngularStep, NumberOfFramesInRotation, TableHeight, TableTraverse ] from RotationInformationSequence." << std::endl;
            }
            // 起始角度
            Float64 startAngle = 0.0;
            if (num_detectors == 1 && item->findAndGetFloat64(DCM_StartAngle, startAngle).good())
            {
                // 如果只有一个探测器，读取起始角度，否则不处理, 加90度，因为默认0度是LPS中的Y轴方向(腹部指向背部)
                detector_start_degree.push_back(precision_t(startAngle + 90));
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
                        rotation_radii.insert(rotation_radii.end(), num_views * num_detectors, precision_t(val));
                    }
                    else
                    {
                        std::cerr << "Failed to parse RadialPosition value at index 0." << std::endl;
                    }
                }
            }
            // 读取旋转方向 顺时针（CW， 减少）或者逆时针（CC，增加）
            if (item->findAndGetOFString(DCM_RotationDirection, rotation_direction).bad())
            {
                std::cerr << "Failed to get RotationDirection from RotationInformationSequence." << std::endl;
            }
        }

        // 2. 读取探测器序列信息
        DcmSequenceOfItems *detectorSeq;
        if (dataset->findAndGetSequence(DCM_DetectorInformationSequence, detectorSeq).bad())
        {
            std::cerr << "Failed to get DetectorInformationSequence." << std::endl;
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
                    image_position_patient_on_z_axis = precision_t(0.0);
                    std::cerr << "ImagePositionPatient does not have enough values, set to 0 as default." << std::endl;
                }
            }
            DcmElement *orientationElem;
            if (item->findAndGetElement(DCM_ImageOrientationPatient, orientationElem).good() && orientationElem->getVR() == EVR_DS)
            {
                const size_t numValues = orientationElem->getNumberOfValues();
                if (numValues >= 6)
                {
                    Float64 val;
                    orientationElem->getFloat64(val, 5);
                    image_orientation_patient_on_z_axis = precision_t(val);
                }
                else
                {
                    image_orientation_patient_on_z_axis = 1.0f;
                    std::cerr << "ImageOrientationPatient does not have enough values, set default value to 1." << std::endl;
                }
            }

            // 读取探测器的旋转起始角度
            double startAngle = 0.0;
            if (num_detectors > 1 && item->findAndGetFloat64(DCM_StartAngle, startAngle).good())
            {
                // 如果有多个探测器，读取每个探测器的起始角度， 加90度，因为默认0度是LPS中的Y轴方向
                detector_start_degree.push_back(precision_t(startAngle + 90));
            }

            // 读取探测器的旋转半径
            DcmElement *radialPosElem;
            if (item->findAndGetElement(DCM_RadialPosition, radialPosElem).good() && radialPosElem->getVR() == EVR_DS)
            {
                const size_t numValues = radialPosElem->getNumberOfValues();
                if (numValues > 1)
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

            // detector_coordinate
            Uint16 size_y, size_z;
            if (dataset->findAndGetUint16(DCM_Rows, size_y).bad() || dataset->findAndGetUint16(DCM_Columns, size_z).bad())
            {
                std::cerr << "Failed to parse image rows and columns." << std::endl;
            }
            Float64 spacing_y, spacing_z;
            if (dataset->findAndGetFloat64(DCM_PixelSpacing, spacing_y, 0).bad() || dataset->findAndGetFloat64(DCM_PixelSpacing, spacing_z, 1).bad())
            {
                std::cerr << "Failed to parse image pixel spacing" << std::endl;
            }
            // 根据image_patient_direction 设置detector_coordinate`
            std::vector<precision_t> direction{1, 0, 0, 0, 1, 0, 0, 0, 1};

            std::vector<size_t> size = {1, size_t(size_y), size_t(size_z)};
            std::vector<precision_t> spacing = {1.0, precision_t(spacing_y), precision_t(spacing_z)};
            std::vector<precision_t> origin = {0.0, precision_t(-spacing_y * size_y / 2 + spacing_y / 2), precision_t(-spacing_z * size_z / 2 + spacing_z / 2)};
            detector_coordinate = {size, spacing, origin, direction};

            detector_end_degree.clear();
            // 设置detector_end_degree
            for (size_t j = 0; j < detector_start_degree.size(); ++j)
            {

                if (rotation_direction == "CC" || rotation_direction == "CCW")
                { // 逆时针角度增加,但是LPS观测方位和NMRECON在Z轴相反，故角度减少
                    detector_end_degree.push_back(precision_t(detector_start_degree[j] - scan_arc));
                }
                else if (rotation_direction == "CW")
                { // 顺时针角度减少，,但是LPS观测方位和NMRECON在Z轴相反，故角度增加
                    detector_end_degree.push_back(precision_t(detector_start_degree[j] + scan_arc));
                }
                else
                {
                    std::cerr << "Invalid RotationDirection value: " << rotation_direction << std::endl;
                }
            }
            table_height = precision_t(table_height_value);
            table_traverse = precision_t(table_traverse_value);
        }
    }
#endif

    void ScanSetting::display() const
    {
        std::cout << "Collimator Shape: " << collimator_shape << "\n";
        std::cout << "Collimator Length: " << collimator_length << "\n";
        std::cout << "Num Projections: " << num_views << "\n";
        std::cout << "Num Detectors: " << num_detectors << "\n";
        std::cout << "Num Rotations: " << num_rotations << "\n";
        std::cout << "Rotation Radii: ";
        for (auto &r : rotation_radii)
            std::cout << r << " ";
        std::cout << "\n";
        std::cout << "Detector Start Degrees: ";
        for (auto &d : detector_start_degree)
            std::cout << d << " ";
        std::cout << "\nDetector End Degrees: ";
        for (auto &d : detector_end_degree)
            std::cout << d << " ";
        std::cout << "\n";
        detector_coordinate.display();
        std::cout << "Image Position Patient on Z Axis: " << image_position_patient_on_z_axis << "\n";
        std::cout << "Image Orientation Patient on Z Axis: " << image_orientation_patient_on_z_axis << "\n";
        std::cout << "Table Height: " << table_height << "\n";
        std::cout << "Table Transverse: " << table_traverse << "\n";
    }

    bool ScanSetting::validate() const
    {
        if (collimator_shape.empty() || num_views <= 0 || num_detectors <= 0 || num_rotations <= 0)
        {
            std::cerr << "Error: Invalid collimator shape, num projections, num detectors or num rotations.\n";
            return false;
        }
        return detector_coordinate.validate();
    }

    void ScanSetting::set_default()
    {
        collimator_shape = "Parallel";
        collimator_length = 10.0;
        num_views = 60;
        rotation_radii = {500.0};
        num_detectors = 2;
        num_rotations = 1;
        detector_start_degree = {0.0, 180.0};
        detector_end_degree = {177.0, 357.0};
        detector_coordinate.set_default();
        image_position_patient_on_z_axis = 0.0;
        image_orientation_patient_on_z_axis = -1.0;
        table_height = 0.0;
        table_traverse = 0.0;
    }

    // 从 JSON 解析 Scanner
    void Scanner::from_json(const json &j)
    {
        name = j.at("name").get<std::string>();
        type = j.at("type").get<std::string>();
        predefined_file = j.at("predefined_file").get<std::string>();
        // TODO: Add predefined file support
        setting.from_json(j.at("setting"));
    }
#ifndef NO_DCMTK
    void Scanner::from_dicom(DcmDataset *dataset)
    {
               // dataset->findAndGetOFString(DCM_ManufacturerModelName, name);
        getDicomString(dataset, DCM_ManufacturerModelName, name);
        // dataset->findAndGetOFString(DCM_Manufacturer, type);
        getDicomString(dataset, DCM_Manufacturer, type);
        predefined_file = std::string("scanner.json");
        setting.from_dicom(dataset);
    }
#endif

    json Scanner::to_json() const
    {
        return json{
            {"type", type},
            {"predefined_file", predefined_file},
            {"name", name},
            {"setting", setting.to_json()}};
    }

    void Scanner::display() const
    {
        std::cout << "Scanner type: " << type << "\n";
        std::cout << "Predefined File: " << predefined_file << "\n";
        std::cout << "Name: " << name << "\n";
        setting.display();
    }

    bool Scanner::validate() const
    {
        if (type.empty() || name.empty())
        {
            std::cerr << "Error: Invalid scanner type or name.\n";
            return false;
        }
        return setting.validate();
    }

    void Scanner::set_default()
    {
        type = "customized";
        predefined_file = "scanner.json";
        name = "mySpect";
        setting.set_default();
    }

    //

    void ScanInfo::from_json(const json &j)
    {
        modality = j.at("modality").get<std::string>();
        acquisition_mode = j.at("acquisition_mode").get<std::string>();
        rawdata_path = j.at("rawdata_path").get<std::string>();
        scanner.from_json(j.at("scanner"));
    }

    json ScanInfo::to_json() const
    {
        return json{
            {"modality", modality},
            {"acquisition_mode", acquisition_mode},
            {"scanner", scanner.to_json()},
            {"rawdata_path", rawdata_path}};
    }

#ifndef NO_DCMTK
    // 根据Dicom信息构造扫描参数
    void ScanInfo::from_dicom(const json &j)
    {
        // 获取 dicom 文件路径
        rawdata_path = j.at("raw_dicom_path").get<std::string>();

        DcmFileFormat fileformat;
        try
        {
            nmrecon::io::DicomProcessor::read_dicom_as_format(rawdata_path, fileformat);
        }
        catch (std::exception &e)
        {
            std::cerr << "Error: Failed to read dicom file: " << e.what() << "\n";
        }

        DcmDataset *dataset = fileformat.getDataset();

        // 获取modality
        // dataset->findAndGetOFString(DCM_Modality, modality);
        getDicomString(dataset, DCM_Modality, modality);
        // 获取 acquisition_mode
        // dataset->findAndGetOFString(DCM_ProtocolName, acquisition_mode);
        getDicomString(dataset, DCM_ProtocolName, acquisition_mode);
        if (acquisition_mode.empty())
            acquisition_mode = j.at("acquisition_mode").get<std::string>();

        // 构造scanner
        scanner.from_dicom(dataset);
    }
#endif

    void ScanInfo::display() const
    {
        std::cout << "Modality: " << modality << "\n";
        std::cout << "Acquisition Mode: " << acquisition_mode << "\n";
        scanner.display();
        std::cout << "Rawdata Path: " << rawdata_path << "\n";
    }

    bool ScanInfo::validate() const
    {
        if (modality.empty() || acquisition_mode.empty())
        {
            std::cerr << "Error: Invalid modality or acquisition mode.\n";
            return false;
        }
        return scanner.validate();
    }
    void ScanInfo::set_default()
    {
        modality = "NM";
        acquisition_mode = "static";
        scanner.set_default();
        rawdata_path = "projection.dcm";
    }

} // namespace io

NMRECON_NAMESPACE_END
