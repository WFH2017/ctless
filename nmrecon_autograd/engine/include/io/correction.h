#pragma once

#include <string>
#include <iostream>
#include "common/types.h"

#include "io/json.h"

NMRECON_NAMESPACE_BEGIN

namespace io
{

    struct OptionalString
    {
        std::string value;
        bool has_value;

        OptionalString() : has_value(false) {}
        OptionalString(const std::string& val) : value(val), has_value(true) {}
    };

    
    struct AttenuationCorrection
    {
        std::string type;
        precision_t energy_keV;
        OptionalString ct_image;

        void from_json(const json& j);
        json to_json() const;
        void display() const;
        bool validate() const;
        void set_default();
    };

    struct ScatterCorrection
    {
        std::string type;
        json scatter_window;
        OptionalString scatter_data_file;

        void from_json(const json& j);
        json to_json() const;
        void display() const;
        bool validate() const;
        void set_default();
    };

    struct PSFCorrection
    {
        std::string type;
        std::string collimator_type;
        std::string radiopharmaceutical;
        std::vector<precision_t> cdr_kernel;
        std::string cdr_kernel_file;
        void from_json(const json& j);
        json to_json() const;
        void display() const;
        bool validate() const;
        void set_default();
    };

    
    struct DecayCorrection
    {
        std::string type;
        std::string radiopharmaceutical;
        precision_t dose;
        precision_t half_life;
        uint64_t injection_time;
        uint64_t scan_time;

        void from_json(const json& j);
        json to_json() const;
        void display() const;
        bool validate() const;
        void set_default();
    };

    
    struct Corrections
    {
        AttenuationCorrection attenuation;
        ScatterCorrection scatter;
        PSFCorrection psf;
        DecayCorrection decay;

        void from_json(const json& j);
        json to_json() const;
        void display() const;
        bool validate() const;
        void set_default();
    };
}

NMRECON_NAMESPACE_END
