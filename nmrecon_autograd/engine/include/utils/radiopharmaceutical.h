#pragma once
#ifndef NMRECON_INCLUDE_CPP_UTILS_RADIOPHARMACEUTICAL_H_
#define NMRECON_INCLUDE_CPP_UTILS_RADIOPHARMACEUTICAL_H_

#include <iostream>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <algorithm> // for std::transform
#include <common/namespace.h>

NMRECON_NAMESPACE_BEGIN
// 定义放射性药物枚举类型
enum class Radiopharmaceutical
{
    Tc_99m_MDP,
    F_18_FDG,
    I_131,
    Lu_177_DOTATATE,
    Y_90_MICROSPHERES,
    Radium_223,
    In_111_OCTREOTIDE,
    Ga_68_DOTATATE,
    Thallium_201,
    Technetium_99m_Sestamibi,
    None
};

std::string getRadiopharmaceuticalName(Radiopharmaceutical drug) {
    switch (drug) {
        case Radiopharmaceutical::Tc_99m_MDP:
            return "Tc-99m-MDP";
        case Radiopharmaceutical::F_18_FDG:
            return "F-18-FDG";
        case Radiopharmaceutical::I_131:
            return "I-131";
        case Radiopharmaceutical::Lu_177_DOTATATE:
            return "Lu-177-DOTATATE";
        case Radiopharmaceutical::Y_90_MICROSPHERES:
            return "Y-90-Microspheres";
        case Radiopharmaceutical::Radium_223:
            return "Radium-223";
        case Radiopharmaceutical::In_111_OCTREOTIDE:
            return "In-111-Octreotide";
        case Radiopharmaceutical::Ga_68_DOTATATE:
            return "Ga-68-DOTATATE";
        case Radiopharmaceutical::Thallium_201:
            return "Thallium-201";
        case Radiopharmaceutical::Technetium_99m_Sestamibi:
            return "Technetium-99m-Sestamibi";
        case Radiopharmaceutical::None:
            return "None";
        default:
            return "Unknown";
    }
}

// 定义一个函数来将字符串转换为枚举类型
Radiopharmaceutical stringToRadiopharmaceutical(const std::string& str) {
    static const std::unordered_map<std::string, Radiopharmaceutical> map = {
        {"Tc-99m-MDP", Radiopharmaceutical::Tc_99m_MDP},
        {"F-18-FDG", Radiopharmaceutical::F_18_FDG},
        {"I-131", Radiopharmaceutical::I_131},
        {"Lu-177-DOTATATE", Radiopharmaceutical::Lu_177_DOTATATE},
        {"Y-90-Microspheres", Radiopharmaceutical::Y_90_MICROSPHERES},
        {"Radium-223", Radiopharmaceutical::Radium_223},
        {"In-111-Octreotide", Radiopharmaceutical::In_111_OCTREOTIDE},
        {"Ga-68-DOTATATE", Radiopharmaceutical::Ga_68_DOTATATE},
        {"Thallium-201", Radiopharmaceutical::Thallium_201},
        {"Technetium-99m-Sestamibi", Radiopharmaceutical::Technetium_99m_Sestamibi},
        {"None", Radiopharmaceutical::None}
    };

    // 替换输入字符串中的空格为连字符
    std::string normalizedStr = str;
    std::replace(normalizedStr.begin(), normalizedStr.end(), ' ', '-');

    // 标准化字符串（统一大小写）
    std::transform(normalizedStr.begin(), normalizedStr.end(), normalizedStr.begin(), ::tolower);

    auto it = map.find(normalizedStr);
    if (it != map.end()) {
        return it->second;
    } else {
        throw std::invalid_argument("Invalid radiopharmaceutical name.");
    }
}
NMRECON_NAMESPACE_END

#endif // NMRECON_INCLUDE_CPP_UTILS_RADIOPHARMACEUTICAL_H_