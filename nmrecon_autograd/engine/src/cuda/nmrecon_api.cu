#include "app/nmrecon_api.h"
#include "scanner/spect.h"
#include "io/scan_info.h"
#include "io/reconstruct_setting.h"
#include "cuda/image3d.h"
#include "cuda/reconstruction/cdr.h"
#include "cuda/reconstruction/distance_driven.h"
#include "cuda/reconstruction/projection.h"

#include <fstream>
#include <memory>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using nmrecon::reconstruction::correction::cdr::GaussianCDR;

namespace {

__global__ void scale_mu_kernel(const float* in, float* out, size_t n, float factor) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmaxf(0.0f, in[idx] * factor);
    }
}

} // namespace

struct NMReconContext {
    nmrecon::io::ScanInfo scan_info;
    nmrecon::io::ReconstructSetting setting;
    std::unique_ptr<nmrecon::scanner::SPECT> scanner;
    nmrecon::common::Coordinate3D image_coord;
    nmrecon::common::Size det_size;
    std::unique_ptr<GaussianCDR> cdr_model;
    bool use_cdr;
    int nx, ny, nz;
    int num_views, det_ny, det_nz;
    size_t image_numel;
    size_t proj_numel;
    int device_id;
    std::unique_ptr<nmrecon::common::Image3D> image_workspace, atten_workspace;
    thrust::device_vector<float> proj_workspace;
};

extern "C" {

void* nmrecon_create(const char* config_json_path, int device_id) {
    std::cerr << "[DEBUG nmrecon_create] called with path=" << (config_json_path ? config_json_path : "NULL") << " device=" << device_id << std::endl;
    if (!config_json_path) return nullptr;
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR nmrecon_create] cudaSetDevice(" << device_id << ") failed: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    
    std::ifstream file(config_json_path);
    if (!file.is_open()) {
        std::cerr << "[ERROR nmrecon_create] failed to open config file: " << config_json_path << std::endl;
        return nullptr;
    }
    
    try {
        json j;
        file >> j;
        auto ctx = new NMReconContext();
        ctx->device_id = device_id;
        ctx->scan_info.from_json(j["ScanInfo"]);
        ctx->setting.from_json(j["ReconstructSetting"]);
        ctx->scanner = std::make_unique<nmrecon::scanner::SPECT>(ctx->scan_info.scanner.setting);
        
        auto img_meta = ctx->setting.output.image_meta;
        ctx->nx = img_meta.size[0];
        ctx->ny = img_meta.size[1];
        ctx->nz = img_meta.size[2];
        ctx->image_numel = size_t(ctx->nx) * ctx->ny * ctx->nz;
        
        ctx->image_coord.set_size(nmrecon::common::Value3<size_t>(ctx->nx, ctx->ny, ctx->nz));
        ctx->image_coord.set_spacing(nmrecon::common::Value3<float>(img_meta.spacing[0], img_meta.spacing[1], img_meta.spacing[2]));
        ctx->image_coord.set_origin(nmrecon::common::Value3<float>(img_meta.origin[0], img_meta.origin[1], img_meta.origin[2]));
        
        auto collimator = ctx->scanner->get_collimator();
        auto det_coord = collimator.get_coordinate_prototype();
        ctx->det_size = det_coord.size;
        ctx->det_ny = det_coord.size.y;
        ctx->det_nz = det_coord.size.z;
        ctx->num_views = ctx->scanner->get_total_views();
        ctx->proj_numel = size_t(ctx->num_views) * ctx->det_ny * ctx->det_nz;
        
        auto cdr_p = ctx->setting.corrections.psf.cdr_kernel;
        ctx->cdr_model = std::make_unique<GaussianCDR>(cdr_p[0], cdr_p[1], cdr_p[2]);
        ctx->use_cdr = (ctx->setting.corrections.psf.type == "cdr");
        
        ctx->image_workspace = std::make_unique<nmrecon::common::Image3D>(ctx->image_coord);
        ctx->atten_workspace = std::make_unique<nmrecon::common::Image3D>(ctx->image_coord);
        ctx->proj_workspace.resize(ctx->proj_numel);
        return static_cast<void*>(ctx);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR nmrecon_create]: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "[ERROR nmrecon_create]: unknown exception" << std::endl;
        return nullptr;
    }
}

void nmrecon_destroy(void* handle) {
    if (handle) {
        auto ctx = static_cast<NMReconContext*>(handle);
        delete ctx;
    }
}

int nmrecon_get_image_size(void* handle, int* nx, int* ny, int* nz) {
    if (!handle) return -1;
    auto ctx = static_cast<NMReconContext*>(handle);
    if (nx) *nx = ctx->nx;
    if (ny) *ny = ctx->ny;
    if (nz) *nz = ctx->nz;
    return 0;
}

int nmrecon_get_proj_size(void* handle, int* num_views, int* det_ny, int* det_nz) {
    if (!handle) return -1;
    auto ctx = static_cast<NMReconContext*>(handle);
    if (num_views) *num_views = ctx->num_views;
    if (det_ny) *det_ny = ctx->det_ny;
    if (det_nz) *det_nz = ctx->det_nz;
    return 0;
}

int nmrecon_forward_device(
    void* handle,
    const float* d_activity,
    const float* d_mu_cm,
    float* d_proj_out
) {
    if (!handle || !d_activity || !d_proj_out) return -1;
    auto ctx = static_cast<NMReconContext*>(handle);
    cudaSetDevice(ctx->device_id);
    
    // Activity image: wrap device data
    auto& image = *ctx->image_workspace;
    cudaMemcpy(thrust::raw_pointer_cast(image.get_data_ref().data()), d_activity, ctx->image_numel*sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Attenuation map: cm^-1 -> mm^-1 (scaled by 0.1)
    auto& atten_map = *ctx->atten_workspace;
    auto& atten_vec = atten_map.get_data_ref();
    if (!d_mu_cm) atten_map.fill(0.0f);
    if (d_mu_cm) {
        int threads = 256;
        int blocks = (ctx->image_numel + threads - 1) / threads;
        scale_mu_kernel<<<blocks, threads>>>(d_mu_cm, thrust::raw_pointer_cast(atten_vec.data()), ctx->image_numel, 0.1f);
        cudaDeviceSynchronize();
    }

    
    // Output projection vector
    auto& proj_vec = ctx->proj_workspace;
    
    // Execute DistanceDriven projection
    if (ctx->use_cdr) {
        nmrecon::reconstruction::projector::distance_driven::cdr_project(
            *ctx->scanner, *ctx->cdr_model, image, atten_map, proj_vec);
    } else {
        nmrecon::reconstruction::projector::distance_driven::atten_project(
            *ctx->scanner, image, atten_map, proj_vec);
    }
    
    // Device-to-device copy into caller's PyTorch tensor buffer
    cudaMemcpy(d_proj_out, thrust::raw_pointer_cast(proj_vec.data()),
               ctx->proj_numel * sizeof(float), cudaMemcpyDeviceToDevice);
    return 0;
}

int nmrecon_backproject_device(
    void* handle,
    const float* d_proj_grad,
    const float* d_mu_cm,
    float* d_act_grad_out
) {
    if (!handle || !d_proj_grad || !d_act_grad_out) return -1;
    auto ctx = static_cast<NMReconContext*>(handle);
    cudaSetDevice(ctx->device_id);
    
    // Output image buffer
    auto& image = *ctx->image_workspace;
    image.fill(0.0f);
    
    // Attenuation map: cm^-1 -> mm^-1
    auto& atten_map = *ctx->atten_workspace;
    auto& atten_vec = atten_map.get_data_ref();
    if (!d_mu_cm) atten_map.fill(0.0f);
    if (d_mu_cm) {
        int threads = 256;
        int blocks = (ctx->image_numel + threads - 1) / threads;
        scale_mu_kernel<<<blocks, threads>>>(d_mu_cm, thrust::raw_pointer_cast(atten_vec.data()), ctx->image_numel, 0.1f);
        cudaDeviceSynchronize();
    }

    
    // Input projection gradient
    auto& proj_vec = ctx->proj_workspace;
    cudaMemcpy(thrust::raw_pointer_cast(proj_vec.data()), d_proj_grad, ctx->proj_numel*sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Execute DistanceDriven backprojection
    if (ctx->use_cdr) {
        nmrecon::reconstruction::projector::distance_driven::cdr_backproject(
            *ctx->scanner, *ctx->cdr_model, image, atten_map, proj_vec);
    } else {
        nmrecon::reconstruction::projector::distance_driven::atten_backproject(
            *ctx->scanner, image, atten_map, proj_vec);
    }
    
    // Device-to-device copy into caller's PyTorch tensor buffer
    const auto& out_vec = image.get_data_ref();
    cudaMemcpy(d_act_grad_out, thrust::raw_pointer_cast(out_vec.data()),
               ctx->image_numel * sizeof(float), cudaMemcpyDeviceToDevice);
    return 0;
}

} // extern "C"
