# 3.28: first non-experimental FILE_SET CXX_MODULES; Ninja >= 1.11.
cmake_minimum_required(VERSION 3.28)

# CMAKE_HOST_UNIX not UNIX: UNIX/WIN32 are unset until project().
if(CMAKE_HOST_UNIX AND SES_AUTODETECT_COMPILER
   AND NOT DEFINED CMAKE_CXX_COMPILER AND "$ENV{CXX}" STREQUAL "")
    # Newest first (LLVM ahead of GCC); first match wins, so extend the top.
    foreach(_cxx IN ITEMS
            clang++-25 clang++-24 clang++-23 clang++-22 clang++-21 clang++-20
            clang++-19 clang++-18 clang++-17 clang++-16
            g++-16 g++-15 g++-14 g++-13 g++-12 clang++ g++)
        # unset(CACHE): find_program caches, else a reused var keeps the first answer.
        unset(_ses_cxx_path CACHE)
        find_program(_ses_cxx_path NAMES ${_cxx})
        if(_ses_cxx_path)
            set(CMAKE_CXX_COMPILER "${_ses_cxx_path}" CACHE FILEPATH "C++ compiler" FORCE)
            string(REPLACE "clang++" "clang" _cc "${_cxx}")
            string(REPLACE "g++" "gcc" _cc "${_cc}")
            unset(_ses_cc_path CACHE)
            find_program(_ses_cc_path NAMES ${_cc})
            if(_ses_cc_path)
                set(CMAKE_C_COMPILER "${_ses_cc_path}" CACHE FILEPATH "C compiler" FORCE)
            endif()
            message(STATUS "SES_AUTODETECT_COMPILER: using ${CMAKE_CXX_COMPILER}")
            break()
        endif()
    endforeach()
endif()

project(ShoeridingerEquationSolver
    VERSION 0.0.1
    DESCRIPTION "From-scratch FDM/TDSE single-electron Schrodinger solver with framework-free Vulkan compute and rendering in an SDL3 window shell"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CMP0155 NEW is the 3.28 default; keep explicit ON so kit presets can't
# silently disable module scanning.
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

# Default to Debug for the TDD loop.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(SES_BUILD_TESTS        "Build unit tests"                          ON)
option(SES_BUILD_APP          "Build the SDL3/Vulkan app (requires SDL3)" ON)
option(SES_BUILD_GPU          "Build the GPU layers solver/viz/scenario + vkcheck (requires vcpkg Vulkan packages)" ON)
option(SES_BUILD_BENCH        "Build the manual micro-benchmark"          ON)
option(SES_ENABLE_AVX2        "Compile with AVX2 (x86-64 CPUs from ~2013)" ON)
option(SES_NATIVE             "Tune for the build host CPU (-march=native)" OFF)
option(SES_AUTODETECT_COMPILER "Pre-project: use the newest clang++/g++ on PATH" OFF)
option(SES_WARNINGS_AS_ERRORS "Treat compiler warnings as errors"         OFF)

# NOTE: never add -ffast-math / /fp:fast -- breaks NaN handling and bitwise
# reproducibility (load-bearing for numerics).
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

# Linked PUBLIC by the core so module BMIs and every importing TU share codegen
# -- a flag mismatch is an ODR trap.
# -ffp-contract=off: -mavx2 -mfma may contract a+b*c into fma(), changing
# rounding and breaking exact-value FP oracles (coord(50) == 0.0).
add_library(ses_arch INTERFACE)
if(NOT MSVC)
    # Disable C99 Annex G NaN/inf fixups on std::complex * and / -- keeps the
    # naive formulas bitwise identical to the oracles. Older Clang lacks the flag;
    # safe to omit (same formula for finite values). MSVC never emits the fixups.
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-fcx-limited-range" SES_HAS_FCX_LIMITED_RANGE)
    if(SES_HAS_FCX_LIMITED_RANGE)
        target_compile_options(ses_arch INTERFACE -fcx-limited-range)
    else()
        check_cxx_compiler_flag("-fcomplex-arithmetic=basic"
                                SES_HAS_FCOMPLEX_ARITH_BASIC)
        if(SES_HAS_FCOMPLEX_ARITH_BASIC)
            target_compile_options(ses_arch INTERFACE -fcomplex-arithmetic=basic)
        else()
            message(STATUS "complex limited-range flag unsupported -- "
                "std::complex keeps Annex G (finite results identical).")
        endif()
    endif()
endif()
if(MSVC)
    if(SES_ENABLE_AVX2 OR SES_NATIVE)
        target_compile_options(ses_arch INTERFACE /arch:AVX2)
        message(STATUS "AVX2 enabled (MSVC /arch:AVX2).")
    endif()
elseif(SES_NATIVE)
    # Pin vector width to 256 (4-wide double) so AVX-512 hosts keep the same
    # reduction order/rounding as the AVX2 build (oracles stay valid).
    target_compile_options(ses_arch INTERFACE
        -march=native -mprefer-vector-width=256 -ffp-contract=off)
    message(STATUS "Native CPU tuning (-march=native, 256-bit vectors).")
elseif(SES_ENABLE_AVX2)
    target_compile_options(ses_arch INTERFACE -mavx2 -mfma -ffp-contract=off)
    message(STATUS "AVX2 enabled (SES_ENABLE_AVX2=ON).")
endif()

add_subdirectory(core)

# Testing must be enabled BEFORE the solver directory registers VkCheck.
if(SES_BUILD_TESTS)
    enable_testing()
endif()

# Gated so the core-only TDD loop configures without vcpkg -- solver
# hard-REQUIREs the vcpkg Vulkan packages.
if(SES_BUILD_GPU)
    include(tools/cmake/ses_slang.cmake)
    include(tools/cmake/ses_vkfft.cmake)
    add_subdirectory(solver)
    add_subdirectory(viz)
    add_subdirectory(scenario)
else()
    message(STATUS "SES_BUILD_GPU=OFF - core-only loop (no solver/viz/scenario/app).")
endif()

if(SES_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(SES_BUILD_BENCH)
    add_subdirectory(bench)
endif()

if(SES_BUILD_APP AND SES_BUILD_GPU)
    find_package(SDL3 CONFIG QUIET)
    if(SDL3_FOUND)
        message(STATUS "SDL3 ${SDL3_VERSION} found - building 'sesolver_app'.")
        add_subdirectory(app)
    else()
        message(WARNING
            "SDL3 not found - skipping the GUI target 'sesolver_app'. "
            "The core library and tests still build. "
            "Install SDL3 (vcpkg manifest) and reconfigure to build the GUI.")
    endif()
endif()
