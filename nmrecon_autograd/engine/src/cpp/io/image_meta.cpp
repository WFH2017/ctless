#include "io/image_meta.h"

NMRECON_NAMESPACE_BEGIN
namespace io
{

    // 从 JSON 解析数据
    void ImageMeta::from_json(const json &j)
    {
        size = j.at("size").get<std::vector<size_t>>();
        spacing = j.at("spacing").get<std::vector<precision_t>>();
        origin = j.at("origin").get<std::vector<precision_t>>();
        direction = j.at("direction").get<std::vector<precision_t>>();
    }

    // 转换为 JSON
    json ImageMeta::to_json() const
    {
        return json{
            {"size", size},
            {"spacing", spacing},
            {"origin", origin},
            {"direction", direction}};
    }

    // 打印内容
    void ImageMeta::display() const
    {
        std::cout << "Detector Coordinate Size: ";
        for (auto &s : size)
            std::cout << s << " ";
        std::cout << "\nSpacing: ";
        for (auto &s : spacing)
            std::cout << s << " ";
        std::cout << "\nOrigin: ";
        for (auto &o : origin)
            std::cout << o << " ";
        std::cout << "\nDirection: ";
        for (auto &d : direction)
            std::cout << d << " ";
        std::cout << "\n";
    }

    // 验证内容
    bool ImageMeta::validate() const
    {
        if (size.size() != 3 || spacing.size() != 3 || origin.size() != 3 || direction.size() != 9)
        {
            std::cerr << "Error: Invalid size, spacing, origin, or direction array size.\n";
            return false;
        }
        return true;
    }

    // 设置默认值
    void ImageMeta::set_default()
    {
        size = {1, 256, 256};
        spacing = {10.0, 2.24, 2.24};
        origin = {0, -285.6, -285.6};
        direction = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    }

} // namespace io

NMRECON_NAMESPACE_END
