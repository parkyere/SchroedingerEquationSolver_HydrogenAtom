# Slang AOT bake (solver/ + viz/): slangc = prebuilt FetchContent tool, no
# runtime/vcpkg dep. Column-major matches host std140 mat4 upload contract.
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

# Shared import-only modules (-I); as bake DEPENDS so an edit rebakes consumers.
set(SES_SLANG_MODULE_DIR "${CMAKE_SOURCE_DIR}/solver/shaders")
file(GLOB SES_SLANG_MODULES CONFIGURE_DEPENDS "${SES_SLANG_MODULE_DIR}/*.slang")
list(FILTER SES_SLANG_MODULES EXCLUDE REGEX "\\.(comp|vert|frag)\\.slang$")

# Stage extension in the srcfile name is a human hint; slangc uses -stage.
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
