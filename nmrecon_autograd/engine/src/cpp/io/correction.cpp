#include <iostream>
#include "io/correction.h"

NMRECON_NAMESPACE_BEGIN

namespace io
{

    void AttenuationCorrection::from_json(const json &j)
    {
        type = j.at("type").get<std::string>();
        energy_keV = j.at("energy_keV").get<precision_t>();
        if (j.contains("ct_image") && !j.at("ct_image").is_null())
        {
            ct_image = OptionalString(j.at("ct_image").get<std::string>());
        }
        else
        {
            ct_image = OptionalString();
        }
    }

    json AttenuationCorrection::to_json() const
    {
        return json{
            {"type", type},
            {"energy_keV", energy_keV},
            {"ct_image", ct_image.has_value ? ct_image.value : "null"}};
    }

    void AttenuationCorrection::display() const
    {
        std::cout << "Attenuation Correction Type: " << type << "\n";
        std::cout << "Energy keV: " << energy_keV << "\n";
        if (ct_image.has_value)
        {
            std::cout << "CT Image: " << ct_image.value << "\n";
        }
        else
        {
            std::cout << "CT Image: null\n";
        }
    }

    bool AttenuationCorrection::validate() const
    {
        if (type.empty() || energy_keV <= 0)
        {
            std::cerr << "Error: Invalid attenuation correction type or energy keV.\n";
            return false;
        }
        return true;
    }

    void AttenuationCorrection::set_default()
    {
        type = "ct_bilinear";
        energy_keV = 140;
        ct_image = OptionalString();
    }

    void ScatterCorrection::from_json(const json &j)
    {
        type = j.at("type").get<std::string>();
        scatter_window = j.at("scatter_window");
        if (j.contains("scatter_data_file") && !j.at("scatter_data_file").is_null())
        {
            scatter_data_file = OptionalString(j.at("scatter_data_file").get<std::string>());
        }
        else
        {
            scatter_data_file = OptionalString();
        }
    }

    json ScatterCorrection::to_json() const
    {
        return json{
            {"type", type},
            {"scatter_window", scatter_window},
            {"scatter_data_file", scatter_data_file.has_value ? scatter_data_file.value : "null"}};
    }

    void ScatterCorrection::display() const
    {
        std::cout << "Scatter Correction Type: " << type << "\n";
        std::cout << "Scatter Window: " << scatter_window.dump() << "\n";
        if (scatter_data_file.has_value)
        {
            std::cout << "Scatter Data File: " << scatter_data_file.value << "\n";
        }
        else
        {
            std::cout << "Scatter Data File: null\n";
        }
    }

    bool ScatterCorrection::validate() const
    {
        if (type.empty())
        {
            std::cerr << "Error: Invalid scatter correction type.\n";
            return false;
        }
        return true;
    }

    void ScatterCorrection::set_default()
    {
        type = "TEW";
        scatter_window = json::object(); // 空的scatter window
        scatter_data_file = OptionalString();
    }

    void PSFCorrection::from_json(const json &j)
    {
        type = j.at("type").get<std::string>();
        collimator_type = j.at("collimator_type").get<std::string>();
        radiopharmaceutical = j.at("radiopharmaceutical").get<std::string>();
        cdr_kernel = j.at("cdr_kernel").get<std::vector<precision_t>>();
        cdr_kernel_file = j.at("cdr_kernel_file").get<std::string>();
    }

    json PSFCorrection::to_json() const
    {
        return json{
            {"type", type},
            {"collimator_type", collimator_type},
            {"radiopharmaceutical", radiopharmaceutical},
            {"cdr_kernel", cdr_kernel},
            {"cdr_kernel_file", cdr_kernel_file}};
    }

    void PSFCorrection::display() const
    {
        std::cout << "PSF Correction Type: " << type << "\n";
        std::cout << "Collimator Type: " << collimator_type << "\n";
        std::cout << "Radiopharmaceutical: " << radiopharmaceutical << "\n";
        std::cout << "CDR kernel: ";
        for (auto &c : cdr_kernel)
            std::cout << c << " ";
        std::cout << "\n";
        std::cout << "CDR Kernel File: " << cdr_kernel_file << "\n";
    }

    bool PSFCorrection::validate() const
    {
        if (type.empty() || collimator_type.empty() || radiopharmaceutical.empty() || cdr_kernel.size() != 3 || cdr_kernel_file.empty())
        {
            std::cerr << "Error: Invalid PSF correction data.\n";
            return false;
        }
        return true;
    }

    void PSFCorrection::set_default()
    {
        type = "cdr";
        collimator_type = "LEHR";
        radiopharmaceutical = "Technetium-99m";
        cdr_kernel = {0.056217, 3.991134, 3.0};
        cdr_kernel_file = "cdr.json";
    }

    // DecayCorrection implementation

    void DecayCorrection::from_json(const json &j)
    {
        type = j.at("type").get<std::string>();
        radiopharmaceutical = j.at("radiopharmaceutical").get<std::string>();
        dose = j.at("dose").get<precision_t>();
        half_life = j.at("half_life").get<precision_t>();
        injection_time = j.at("injection_time").get<uint64_t>();
        scan_time = j.at("scan_time").get<uint64_t>();
    }

    json DecayCorrection::to_json() const
    {
        return json{
            {"type", type},
            {"radiopharmaceutical", radiopharmaceutical},
            {"dose", dose},
            {"half_life", half_life},
            {"injection_time", injection_time},
            {"scan_time", scan_time}};
    }

    void DecayCorrection::display() const
    {
        std::cout << "Decay Correction Type: " << type << "\n";
        std::cout << "Radiopharmaceutical: " << radiopharmaceutical << "\n";
        std::cout << "Dose: " << dose << "\n";
        std::cout << "Half Life: " << half_life << "\n";
        std::cout << "Injection Time: " << injection_time << "\n";
        std::cout << "Scan Time: " << scan_time << "\n";
    }

    bool DecayCorrection::validate() const
    {
        if (type.empty() || radiopharmaceutical.empty() || dose <= 0 || half_life <= 0)
        {
            std::cerr << "Error: Invalid decay correction data.\n";
            return false;
        }
        return true;
    }

    void DecayCorrection::set_default()
    {
        type = "exponential";
        radiopharmaceutical = "Technetium-99m";
        dose = 192000000;                // 例：192 MBq
        half_life = 6000.0;              // 例：6000秒
        injection_time = 20240908165823; // 示例时间戳
        scan_time = 20240908173032;      // 示例时间戳
    }

    // Corrections implementation

    void Corrections::from_json(const json &j)
    {
        attenuation.from_json(j.at("attenuation"));
        scatter.from_json(j.at("scatter"));
        psf.from_json(j.at("psf"));
        decay.from_json(j.at("decay"));
    }

    json Corrections::to_json() const
    {
        return json{
            {"attenuation", attenuation.to_json()},
            {"scatter", scatter.to_json()},
            {"psf", psf.to_json()},
            {"decay", decay.to_json()}};
    }

    void Corrections::display() const
    {
        std::cout << "Corrections: \n";
        attenuation.display();
        scatter.display();
        psf.display();
        decay.display();
    }

    bool Corrections::validate() const
    {
        return attenuation.validate() && scatter.validate() && psf.validate() && decay.validate();
    }

    void Corrections::set_default()
    {
        attenuation.set_default();
        scatter.set_default();
        psf.set_default();
        decay.set_default();
    }
}
NMRECON_NAMESPACE_END
