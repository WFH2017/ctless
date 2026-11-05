
#include "io/reconstruct_setting.h"

NMRECON_NAMESPACE_BEGIN
namespace io
{

    void ReconstructSetting::from_json(const json &j)
    {
        reconstruction.from_json(j.at("reconstruction"));
        corrections.from_json(j.at("corrections"));
        output.from_json(j.at("output"));
    }

    json ReconstructSetting::to_json() const
    {
        return json{
            {"reconstruction", reconstruction.to_json()},
            {"corrections", corrections.to_json()},
            {"output", output.to_json()}};
    }

    void ReconstructSetting::display() const
    {
        std::cout << "Reconstruct Setting: \n";
        reconstruction.display();
        corrections.display();
        output.display();
    }

    bool ReconstructSetting::validate() const
    {
        return reconstruction.validate() && corrections.validate() && output.validate();
    }

    void ReconstructSetting::set_default()
    {
        reconstruction.set_default();
        corrections.set_default();
        output.set_default();
    }
}
NMRECON_NAMESPACE_END