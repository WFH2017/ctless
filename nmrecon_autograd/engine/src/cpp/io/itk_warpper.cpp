
#include "io/itk_warpper.h" // 不包含ITK，只有函数声明
#include "io/itk.h"
NMRECON_NAMESPACE_BEGIN
namespace io
{

     std::vector<uint16_t> load_spect_nifty_cpu(const std::string& path)
    {
        std::vector<uint16_t> buffer;
        ITKProcessor::load_spect_nifty(path, buffer);
        return buffer;
    }
}

NMRECON_NAMESPACE_END