#pragma once
#ifndef NMRECON_INCLUDE_CPP_APP_BASE_H_
#define NMRECON_INCLUDE_CPP_APP_BASE_H_

#include <common/namespace.h>
#include <string>

NMRECON_NAMESPACE_BEGIN
namespace app
{

    class BaseApp
    {
    public:
        BaseApp() = default;
        virtual ~BaseApp() = default;
        virtual int run() const = 0;
        virtual void help() const = 0;
        virtual int parse(const std::string& config_file) = 0;

    };
}
NMRECON_NAMESPACE_END

#endif // NMRECON_INCLUDE_CPP_APP_BASE_H_