#pragma once
#ifndef NMRECON_INCLUDE_CPP_IO_OUTPUT_H_
#define NMRECON_INCLUDE_CPP_IO_OUTPUT_H_

#include "common/namespace.h"
#include "io/image_meta.h"

NMRECON_NAMESPACE_BEGIN
namespace io
{

    struct Output
    {
        std::string path;
        std::string format;
        int save_interval;
        ImageMeta image_meta;

        void from_json(const json &j);

        json to_json() const;
        void display() const;
        bool validate() const;

        void set_default();
    };

} // namespace io

NMRECON_NAMESPACE_END

#endif // NMRECON_INCLUDE_CPP_IO_OUTPUT_H_