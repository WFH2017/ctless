#include "io/output.h"

NMRECON_NAMESPACE_BEGIN
namespace io
{
    void Output::from_json(const json &j)
    {
        path = j.at("path").get<std::string>();
        format = j.at("format").get<std::string>();
        save_interval = j.at("save_interval").get<int>();
        image_meta.from_json(j.at("image_meta"));
    }
    json Output::to_json() const
    {
        return json{
            {"path", path},
            {"format", format},
            {"save_interval", save_interval},
            {"image_meta", image_meta.to_json()}};
    }

    void Output::display() const
    {
        std::cout << "Output Settings:\n";
        std::cout << "  Path: " << path << "\n";
        std::cout << "  Format: " << format << "\n";
        std::cout << "  Save Interval: " << save_interval << "\n";
        image_meta.display();
    }

    bool Output::validate() const
    {
        if (path.empty() || format.empty() || save_interval <= 0)
        {
            std::cerr << "Error: Output settings validation failed. Path, format, or save_interval are incorrect.\n";
            return false;
        }
        return image_meta.validate();
    }

    void Output::set_default()
    {
        path = "reconpath";
        format = "nii";
        save_interval = 5;
        image_meta.set_default();
    }

} // namespace io

NMRECON_NAMESPACE_END