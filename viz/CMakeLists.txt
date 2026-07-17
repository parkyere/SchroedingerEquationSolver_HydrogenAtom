# ---------------------------------------------------------------------------
# ses_viz: the raw-Vulkan visualization library -- scene renderer (volume /
# mesh / slice / flow / HDR-bloom compose) and the swapchain presenter. Rides
# the solver's device; knows NOTHING about SDL or ImGui (the presenter takes
# an opaque UI-record callback). Consumers import ses.vk.render /
# ses.vk.present / ses.vk.render_blobs.
# ---------------------------------------------------------------------------
set(_ses_viz_spv_dir "${CMAKE_CURRENT_BINARY_DIR}/spv")
file(MAKE_DIRECTORY "${_ses_viz_spv_dir}")
set(_ses_viz_hdrs "")
foreach(_k IN ITEMS accum bloom_down bloom_up compose particles
        occupancy occ_dilate shadow)
    ses_bake_shader(_ses_viz_hdrs "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
                    "${_ses_viz_spv_dir}" "${_k}.comp" compute "${_k}")
endforeach()
foreach(_k IN ITEMS mesh.vert mesh.frag volume.vert volume.frag
        slice.vert slice.frag
        flow.vert flow.frag overlay.vert overlay.frag blit.vert blit.frag)
    string(REPLACE "." "_" _x "${_k}")
    if(_k MATCHES "\\.vert$")
        set(_stage vertex)
    else()
        set(_stage fragment)
    endif()
    ses_bake_shader(_ses_viz_hdrs "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
                    "${_ses_viz_spv_dir}" "${_k}" "${_stage}" "${_x}")
endforeach()
add_custom_target(ses_viz_spv DEPENDS ${_ses_viz_hdrs})

# Sources are the CXX_MODULES file set -- each interface unit compiles to a
# real object, so the archive needs no placeholder TU.
add_library(ses_viz STATIC)
target_sources(ses_viz
    PUBLIC FILE_SET CXX_MODULES FILES
        src/vk_render.ixx
        src/vk_present.ixx
        src/render_blobs.ixx)
target_compile_features(ses_viz PUBLIC cxx_std_20)
add_dependencies(ses_viz ses_viz_spv)
# PUBLIC spv dir: the shell's own swapchain pass textually includes the blit
# pair (blit_vert_spv.h / blit_frag_spv.h) from here.
target_include_directories(ses_viz PUBLIC "${_ses_viz_spv_dir}")
target_link_libraries(ses_viz
    PUBLIC
        ses_solver
    PRIVATE
        ses_warnings)
