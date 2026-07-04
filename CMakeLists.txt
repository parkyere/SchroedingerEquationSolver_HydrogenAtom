cmake_minimum_required(VERSION 3.21)

project(ShoeridingerEquationSolver
    VERSION 0.0.1
    DESCRIPTION "From-scratch FDM/TDSE single-electron Schrodinger solver with hand-written OpenGL visualization"
    LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Language standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Single-config generators: default to Debug for the TDD loop.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# compile_commands.json for clangd / tooling.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(SES_BUILD_TESTS        "Build unit tests"                          ON)
option(SES_BUILD_APP          "Build the Qt/OpenGL app (requires Qt6)"    ON)
option(SES_BUILD_BENCH        "Build the manual micro-benchmark"          ON)
option(SES_ENABLE_AVX2        "Compile with AVX2 (x86-64 CPUs from ~2013)" ON)
option(SES_WARNINGS_AS_ERRORS "Treat compiler warnings as errors"         OFF)

# ---------------------------------------------------------------------------
# Shared warning flags (INTERFACE target).
# NOTE: never enable -ffast-math / /fp:fast - it breaks NaN handling and
# bitwise reproducibility, which are load-bearing for numerical correctness.
# ---------------------------------------------------------------------------
add_library(ses_warnings INTERFACE)
if(MSVC)
    target_compile_options(ses_warnings INTERFACE /W4 /permissive- /EHsc)
    if(SES_WARNINGS_AS_ERRORS)
        target_compile_options(ses_warnings INTERFACE /WX)
    endif()
else()
    target_compile_options(ses_warnings INTERFACE -Wall -Wextra -Wpedantic)
    if(SES_WARNINGS_AS_ERRORS)
        target_compile_options(ses_warnings INTERFACE -Werror)
    endif()
endif()

# ---------------------------------------------------------------------------
# CPU architecture flags (INTERFACE target, linked PUBLIC by the core so every
# TU that compiles the header-only hot loops gets it).
#
# NOTE -ffp-contract=off: with bare -mavx2 -mfma the compiler may contract
# a + b*c into fma(b, c, a), changing rounding and breaking the exact-value
# FP oracles the tests pin (e.g. symmetric-grid coord(50) == 0.0). Contraction
# off keeps scalar semantics BITWISE identical to the SSE2 build -- we take
# only the 4-wide double vectorization.
# ---------------------------------------------------------------------------
add_library(ses_arch INTERFACE)
if(SES_ENABLE_AVX2)
    if(MSVC)
        target_compile_options(ses_arch INTERFACE /arch:AVX2)
    else()
        target_compile_options(ses_arch INTERFACE -mavx2 -mfma -ffp-contract=off)
    endif()
    message(STATUS "AVX2 enabled (SES_ENABLE_AVX2=ON).")
endif()

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
# Pure, dependency-free numerical core. Always built.
add_subdirectory(core)

if(SES_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(SES_BUILD_BENCH)
    add_subdirectory(bench)
endif()

# GUI shell. Auto-skipped (with a warning) when Qt6 is not installed, so the
# core library and the TDD test loop never depend on a Qt toolchain.
if(SES_BUILD_APP)
    find_package(Qt6 COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets QUIET)
    if(Qt6_FOUND)
        message(STATUS "Qt6 ${Qt6_VERSION} found - building 'sesolver_app'.")
        add_subdirectory(app)
    else()
        message(WARNING
            "Qt6 not found - skipping the GUI target 'sesolver_app'. "
            "The core library and tests still build. "
            "Install Qt6 (Core Gui Widgets OpenGL OpenGLWidgets) and reconfigure to build the GUI.")
    endif()
endif()
