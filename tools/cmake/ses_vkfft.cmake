# VkFFT is header-only + compiles shaders via glslang at runtime -> find/link glslang.
# vcpkg buries the C header under glslang/Include -> PATH_SUFFIXES below.
# FOOTGUN: never add the vulkan-1 import lib -- thunks collide with volk PFN
# globals -> volkInitialize crash.
# SES_HAVE_VKFFT gates Engine members -- apply to lib AND every TU (single-BMI
# rule: divergent defines bake divergent engines).
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
