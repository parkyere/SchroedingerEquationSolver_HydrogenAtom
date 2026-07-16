# ---------------------------------------------------------------------------
# VkFFT: header-only FFT + its runtime shader compiler (glslang). Optional --
# when absent the engine compiles without SES_HAVE_VKFFT and stays on the
# hand-rolled line FFT. VkFFT includes the bare 'glslang_c_interface.h';
# vcpkg installs it under glslang/Include/, so that directory must be on the
# include path. NOTE: every ses_vk target lives in the volk world
# (VK_NO_PROTOTYPES + PFN globals) -- NEVER link the vulkan-1 IMPORT library
# into one: its code thunks collide with volk's identically named function-
# pointer variables and volkInitialize crashes on startup.
# The SES_HAVE_VKFFT define gates Engine MEMBERS, so it must be applied
# consistently to the module lib AND every TU with its own #ifdef blocks
# (single-BMI rule: divergent defines would bake divergent engines).
# ---------------------------------------------------------------------------
find_package(glslang CONFIG QUIET)
find_path(SES_GLSLANG_C_INCLUDE_DIR glslang_c_interface.h
          PATH_SUFFIXES glslang/Include)
if(TARGET glslang::glslang AND SES_GLSLANG_C_INCLUDE_DIR)
    set(SES_VKFFT_AVAILABLE ON)
    message(STATUS "VkFFT enabled (engine step-body 3D transforms).")
else()
    set(SES_VKFFT_AVAILABLE OFF)
    message(STATUS "VkFFT unavailable -- engine keeps the hand-rolled FFT.")
endif()
function(ses_enable_vkfft target)
    if(SES_VKFFT_AVAILABLE)
        target_compile_definitions(${target} PRIVATE SES_HAVE_VKFFT=1 VKFFT_BACKEND=0)
        target_include_directories(${target} SYSTEM PRIVATE
            "${SES_GLSLANG_C_INCLUDE_DIR}")
        target_link_libraries(${target} PRIVATE
            glslang::glslang
            glslang::SPIRV
            glslang::glslang-default-resource-limits)
    endif()
endfunction()
