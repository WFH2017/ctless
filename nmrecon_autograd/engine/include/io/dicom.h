/**
 * @file dicom.h
 * @brief DICOM file IO
 */

#pragma once


#include <iostream>
#include <dcmtk/dcmdata/dctk.h>
#include <dcmtk/dcmimgle/dcmimage.h>
#include <dcmtk/dcmdata/dcuid.h>    // for UID functions
#include <dcmtk/dcmdata/dcfilefo.h> // for DcmFileFormat
#include <dcmtk/dcmdata/dcdeftag.h>
#include <dcmtk/dcmdata/dcmetinf.h>
#include <dcmtk/dcmdata/dcvrlo.h>
#include <dcmtk/dcmdata/dcvrui.h>
#include "common/namespace.h"
#include "common/basic_type.h"

NMRECON_NAMESPACE_BEGIN
namespace io
{
    struct DicomProcessor
    {
        static bool read_dicom_as_format(const std::string &filepath, DcmFileFormat &fileFormat);

        static bool transfer_dicom_meta(DcmFileFormat &src, DcmFileFormat &dst, const char *newSOPInstanceUID);
        
        static bool load_spect_raw(DcmDataset *dataset, std::vector<uint16_t> &imageOut);

        static bool set_output_image(DcmDataset *destDataset,
                                    const std::vector<uint16_t> &image_vector,
                                    const nmrecon::common::Coordinate3D &image_meta);

        static bool save_format_to_dicom(const std::string &filepath,
                                         DcmFileFormat &fileFormat,
                                         const E_TransferSyntax &xfer);

        static bool save_dicom(const std::string &filepath, DcmFileFormat &fileFormat);

        // new methods for loading projections
        static bool build_trajectory(); 
        static bool compute_recon_fov(); //reconstruct fov.

        static bool load_static_raw(DcmDataset *dataset, std::vector<uint16_t> &projection_values);
        static bool load_dynamic_raw(DcmDataset *dataset, std::vector<uint16_t> &projection_values);
        
        // load SPECT projection data and set the metadata for reconstruction

        static bool load_spect_projection(const std::string &filepath);
        
        
    };
}
NMRECON_NAMESPACE_END