#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Create an NMRecon reconstruction context from a recon.json config file
// Returns an opaque context pointer, or NULL on error
void* nmrecon_create(const char* config_json_path, int device_id);

// Free an NMRecon context
void nmrecon_destroy(void* handle);

// Query dimensions
int nmrecon_get_image_size(void* handle, int* nx, int* ny, int* nz);
int nmrecon_get_proj_size(void* handle, int* num_views, int* det_ny, int* det_nz);

// Direct GPU-pointer Forward Project (Zero disk I/O, device-to-device)
// d_activity: [nx, ny, nz] float32 on CUDA device
// d_mu_cm:    [nx, ny, nz] float32 in cm^-1 on CUDA device (NULL for no attenuation)
// d_proj_out: [num_views, det_ny, det_nz] float32 on CUDA device
int nmrecon_forward_device(
    void* handle,
    const float* d_activity,
    const float* d_mu_cm,
    float* d_proj_out
);

// Adjoint backproject from GPU pointer to GPU pointer
// d_proj_grad:    [num_views, det_ny, det_nz] float32 on CUDA device
// d_mu_cm:        [nx, ny, nz] float32 in cm^-1 on CUDA device (NULL for no attenuation)
// d_act_grad_out: [nx, ny, nz] float32 on CUDA device
int nmrecon_backproject_device(
    void* handle,
    const float* d_proj_grad,
    const float* d_mu_cm,
    float* d_act_grad_out
);

#ifdef __cplusplus
}
#endif
