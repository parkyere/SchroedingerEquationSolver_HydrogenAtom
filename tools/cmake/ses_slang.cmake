# ---------------------------------------------------------------------------
# Slang AOT toolchain (shared by solver/ and viz/). Every ses_vk compute
# kernel and render shader is a Slang source baked to plain SPIR-V by slangc
# at build time and embedded as a C array via tools/cmake/bin2h.cmake
# (symbol k_<x>_spv). slangc is a prebuilt BUILD-TIME tool (FetchContent) --
# it adds no runtime or vcpkg dependency; matrices are column-major to match
# the host std140 mat4 upload.
# ---------------------------------------------------------------------------
include(FetchContent)
set(SES_SLANG_VERSION "2026.13.1")
if(WIN32)
    set(_ses_slang_asset "slang-${SES_SLANG_VERSION}-windows-x86_64.zip")
else()
    set(_ses_slang_asset "slang-${SES_SLANG_VERSION}-linux-x86_64.zip")
endif()
FetchContent_Declare(ses_slang
    URL "https://github.com/shader-slang/slang/releases/download/v${SES_SLANG_VERSION}/${_ses_slang_asset}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(ses_slang)
find_program(SES_SLANGC slangc
    HINTS "${ses_slang_SOURCE_DIR}/bin"
    NO_DEFAULT_PATH REQUIRED)
message(STATUS "Slang AOT compiler: ${SES_SLANGC}")

# Shared Slang modules (complex, ylm, grid, vec ...) live in solver/shaders:
# they are physics math, `import`-ed by both solver kernels and viz shaders
# (via -I), never baked as entry points; listed as bake deps so editing a
# module rebakes its consumers.
set(SES_SLANG_MODULE_DIR "${CMAKE_SOURCE_DIR}/solver/shaders")
file(GLOB SES_SLANG_MODULES CONFIGURE_DEPENDS "${SES_SLANG_MODULE_DIR}/*.slang")
list(FILTER SES_SLANG_MODULES EXCLUDE REGEX "\\.(comp|vert|frag)\\.slang$")

# Bake one Slang shader (<srcdir>/<srcfile>.slang -> <spvdir>/<x>_spv.h,
# symbol k_<x>_spv) and append the header to the list variable <hdrs_var>
# (caller scope). The stage extension in the source name is a human hint;
# slangc takes -stage.
function(ses_bake_shader hdrs_var srcdir spvdir _srcfile _stage _x)
    set(_slang "${srcdir}/${_srcfile}.slang")
    set(_spv "${spvdir}/${_x}.spv")
    set(_hdr "${spvdir}/${_x}_spv.h")
    add_custom_command(
        OUTPUT "${_hdr}"
        BYPRODUCTS "${_spv}"
        COMMAND "${SES_SLANGC}" "${_slang}"
                -I "${SES_SLANG_MODULE_DIR}"
                -target spirv -entry main -stage ${_stage}
                -matrix-layout-column-major -o "${_spv}"
        COMMAND "${CMAKE_COMMAND}" "-DIN=${_spv}" "-DOUT=${_hdr}"
                "-DNAME=k_${_x}_spv"
                -P "${CMAKE_SOURCE_DIR}/tools/cmake/bin2h.cmake"
        DEPENDS "${_slang}" ${SES_SLANG_MODULES}
                "${CMAKE_SOURCE_DIR}/tools/cmake/bin2h.cmake"
        COMMENT "slangc: ${_srcfile}.slang -> ${_x}_spv.h"
        VERBATIM)
    set(_hdrs "${${hdrs_var}}")
    list(APPEND _hdrs "${_hdr}")
    set(${hdrs_var} "${_hdrs}" PARENT_SCOPE)
endfunction()
