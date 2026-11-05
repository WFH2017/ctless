#pragma once

#include "common/namespace.h"
#include "common/types.h"
// #include "io/parser.h"
#include "io/image_meta.h"

#include "io/json.h"
#ifndef NO_DCMTK
#include "io/dicom.h"
#else
#include <cstdint>
using Uint16 = uint16_t;
#endif

NMRECON_NAMESPACE_BEGIN
namespace io
{

    struct ScanSetting
    {
        std::string collimator_shape;
        precision_t collimator_length;
        Uint16 num_views;
        Uint16 num_detectors;
        Uint16 num_rotations;

        precision_t image_position_patient_on_z_axis;
        precision_t image_orientation_patient_on_z_axis;
        precision_t table_height;
        precision_t table_traverse;

        std::vector<precision_t> rotation_radii;
        std::vector<precision_t> detector_start_degree;
        std::vector<precision_t> detector_end_degree;
        ImageMeta detector_coordinate;

        void from_json(const json &j);
        json to_json() const;
        
#ifndef NO_DCMTK
        void from_dicom(DcmDataset *dataset);
#endif

        void display() const;
        bool validate() const;
        void set_default();
    };

    struct Scanner
    {
        std::string type;
        std::string predefined_file;
        std::string name;
        ScanSetting setting;

        void from_json(const json &j);
        
        json to_json() const;

#ifndef NO_DCMTK
        void from_dicom(DcmDataset *dataset);
#endif

        void display() const;

        bool validate() const;

        void set_default();
    };

    struct ScanInfo
    {
        std::string modality;
        std::string acquisition_mode;
        std::string rawdata_path;
        Scanner scanner;

        void from_json(const json &j);

        json to_json() const;

#ifndef NO_DCMTK
        // 根据Dicom信息构造扫描参数
        void from_dicom(const json &j);
#endif

        void display() const;

        bool validate() const;

        void set_default();
    };

} // namespace io
NMRECON_NAMESPACE_END
