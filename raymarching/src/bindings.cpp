#include <torch/extension.h>

#include "raymarching.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // utils
    m.def("flatten_rays", &flatten_rays, "flatten_rays (CUDA)");
    m.def("packbits", &packbits, "packbits (CUDA)");
    m.def("packbits_triplane", &packbits_triplane, "packbits_triplane (CUDA)");
    m.def("near_far_from_aabb", &near_far_from_aabb, "near_far_from_aabb (CUDA)");
    m.def("sph_from_ray", &sph_from_ray, "sph_from_ray (CUDA)");
    m.def("morton3D", &morton3D, "morton3D (CUDA)");
    m.def("morton3D_invert", &morton3D_invert, "morton3D_invert (CUDA)");
    m.def("morton2D", &morton2D, "morton2D (CUDA)");
    m.def("morton2D_invert", &morton2D_invert, "morton2D_invert (CUDA)");
    // train
    m.def("march_rays_train", &march_rays_train, "march_rays_train (CUDA)");
    m.def("march_rays_triplane_train", &march_rays_triplane_train, "march_rays_triplane_train (CUDA)");
    m.def("composite_rays_train_forward", &composite_rays_train_forward, "composite_rays_train_forward (CUDA)");
    m.def("composite_rays_train_neus_forward", &composite_rays_train_neus_forward, "composite_rays_train_neus_forward (CUDA)");
    m.def("composite_rays_train_hybrid_forward", &composite_rays_train_hybrid_forward, "composite_rays_train_hybrid_forward (CUDA)");
    m.def("composite_rays_train_backward", &composite_rays_train_backward, "composite_rays_train_backward (CUDA)");
    m.def("composite_rays_train_neus_backward", &composite_rays_train_neus_backward, "composite_rays_train_neus_backward (CUDA)");
    m.def("composite_rays_train_hybrid_backward", &composite_rays_train_hybrid_backward, "composite_rays_train_hybrid_backward (CUDA)");
    // infer
    m.def("march_rays", &march_rays, "march rays (CUDA)");
    m.def("composite_rays", &composite_rays, "composite rays (CUDA)");
}