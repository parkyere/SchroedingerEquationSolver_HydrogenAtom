module;
#include <mesh_vert_spv.h>
#include <mesh_frag_spv.h>
#include <volume_vert_spv.h>
#include <volume_frag_spv.h>
#include <slice_vert_spv.h>
#include <slice_frag_spv.h>
#include <accum_spv.h>
#include <bloom_down_spv.h>
#include <bloom_up_spv.h>
#include <compose_spv.h>
#include <particles_spv.h>
#include <occupancy_spv.h>
#include <occ_dilate_spv.h>
#include <shadow_spv.h>
#include <flow_vert_spv.h>
#include <flow_frag_spv.h>
export module ses.vk.render_blobs;
export import ses.vk.render;


// The viz library's embedded SPIR-V blobs: every render/post shader the
// raw-Vulkan scene renderer runs, baked offline from viz/shaders/.


export namespace ses_vk {

inline RenderKernels render_blobs() {
    RenderKernels r;
    r.mesh_vert = k_mesh_vert_spv;
    r.mesh_vert_size = k_mesh_vert_spv_size;
    r.mesh_frag = k_mesh_frag_spv;
    r.mesh_frag_size = k_mesh_frag_spv_size;
    r.volume_vert = k_volume_vert_spv;
    r.volume_vert_size = k_volume_vert_spv_size;
    r.volume_frag = k_volume_frag_spv;
    r.volume_frag_size = k_volume_frag_spv_size;
    r.slice_vert = k_slice_vert_spv;
    r.slice_vert_size = k_slice_vert_spv_size;
    r.slice_frag = k_slice_frag_spv;
    r.slice_frag_size = k_slice_frag_spv_size;
    r.accum = k_accum_spv;
    r.accum_size = k_accum_spv_size;
    r.bloom_down = k_bloom_down_spv;
    r.bloom_down_size = k_bloom_down_spv_size;
    r.bloom_up = k_bloom_up_spv;
    r.bloom_up_size = k_bloom_up_spv_size;
    r.compose = k_compose_spv;
    r.compose_size = k_compose_spv_size;
    r.particles = k_particles_spv;
    r.particles_size = k_particles_spv_size;
    r.occupancy = k_occupancy_spv;
    r.occupancy_size = k_occupancy_spv_size;
    r.occ_dilate = k_occ_dilate_spv;
    r.occ_dilate_size = k_occ_dilate_spv_size;
    r.shadow = k_shadow_spv;
    r.shadow_size = k_shadow_spv_size;
    r.flow_vert = k_flow_vert_spv;
    r.flow_vert_size = k_flow_vert_spv_size;
    r.flow_frag = k_flow_frag_spv;
    r.flow_frag_size = k_flow_frag_spv_size;
    return r;
}

}  // namespace ses_vk
