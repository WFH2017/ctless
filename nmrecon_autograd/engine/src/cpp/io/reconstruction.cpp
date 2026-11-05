#include "io/reconstruction.h"

NMRECON_NAMESPACE_BEGIN
namespace io
{

    void Reconstruction::from_json(const json &j)
    {
        type = j.at("type").get<std::string>();
        num_iterations = j.at("num_iterations").get<int>();
        num_subsets = j.at("num_subsets").get<int>();
        projector = j.at("projector").get<std::string>();
        gaussian_sigma = j.at("gaussian_sigma").get<precision_t>();
    }

    json Reconstruction::to_json() const
    {
        return json{
            {"type", type},
            {"num_iterations", num_iterations},
            {"num_subsets", num_subsets},
            {"projector", projector},
            {"gaussian_sigma", gaussian_sigma}};
    }

    void Reconstruction::display() const
    {
        std::cout << "Reconstruction Type: " << type << "\n";
        std::cout << "Num Iterations: " << num_iterations << "\n";
        std::cout << "Num Subsets: " << num_subsets << "\n";
        std::cout << "Projector: " << projector << "\n";
        std::cout << "Gaussian Sigma: " << gaussian_sigma << "\n";
    }
    bool Reconstruction::validate() const
    {
        if (type.empty() || num_iterations <= 0 || num_subsets <= 0)
        {
            std::cerr << "Error: Invalid reconstruction type, number of iterations, or subsets.\n";
            return false;
        }
        return true;
    }

    void Reconstruction::set_default()
    {
        type = "OSEM";
        num_iterations = 30;
        num_subsets = 1;
        projector = "DistanceDriven";
        gaussian_sigma = 0;
    }

} // namespace io

NMRECON_NAMESPACE_END