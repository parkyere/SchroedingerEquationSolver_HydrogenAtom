find_package(SDL3 CONFIG REQUIRED)
find_package(volk CONFIG REQUIRED)  # imported targets are directory-scoped
find_package(Boost REQUIRED COMPONENTS program_options)

# Vendored ImGui, NOT vcpkg port: its binding links vulkan-1 import lib,
# forbidden with volk (tools/cmake/ses_vkfft.cmake). USE_VOLK rides volk's loader.
set(SES_IMGUI_DIR "${CMAKE_SOURCE_DIR}/external/imgui")
if(NOT EXISTS "${SES_IMGUI_DIR}/imgui.cpp")
    message(FATAL_ERROR "external/imgui submodule missing -- run "
        "'git submodule update --init external/imgui'.")
endif()
add_library(ses_imgui STATIC
    "${SES_IMGUI_DIR}/imgui.cpp"
    "${SES_IMGUI_DIR}/imgui_draw.cpp"
    "${SES_IMGUI_DIR}/imgui_tables.cpp"
    "${SES_IMGUI_DIR}/imgui_widgets.cpp"
    "${SES_IMGUI_DIR}/backends/imgui_impl_sdl3.cpp"
    "${SES_IMGUI_DIR}/backends/imgui_impl_vulkan.cpp")
target_include_directories(ses_imgui SYSTEM PUBLIC
    "${SES_IMGUI_DIR}" "${SES_IMGUI_DIR}/backends")
target_compile_definitions(ses_imgui PUBLIC IMGUI_IMPL_VULKAN_USE_VOLK)
target_link_libraries(ses_imgui PUBLIC volk::volk SDL3::SDL3)
target_compile_features(ses_imgui PRIVATE cxx_std_20)

add_executable(sesolver_app
    src/main.cpp
)
target_sources(sesolver_app
    PRIVATE FILE_SET CXX_MODULES FILES
        src/imgui_ui.ixx
        src/scheduler.ixx)
# WIN32 subsystem (optimized builds) drops the console; Debug keeps it. SDL_main.h = WinMain.
set_target_properties(sesolver_app PROPERTIES
    WIN32_EXECUTABLE "$<CONFIG:Release,RelWithDebInfo,MinSizeRel>"
)
target_link_libraries(sesolver_app
    PRIVATE
        ses_scenario
        ses_viz
        SDL3::SDL3
        ses_imgui
        ses_warnings
        Boost::program_options
)
target_compile_features(sesolver_app PRIVATE cxx_std_20)
# _CRT_SECURE_NO_WARNINGS: MSVC flags std::getenv; SES_VK_VALIDATION read once at boot, race-free.
if(MSVC)
    target_compile_definitions(sesolver_app PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Opt-in: a full arc sweep is a multi-minute GPU battery, too heavy for the red/green loop.
option(SES_ARC_TESTS "Register the --selftest-* arcs as ctest entries (slow, GPU)" OFF)
if(SES_ARC_TESTS)
    foreach(_arc IN ITEMS anderson benzene billiard bloch bouncer carpet
            cascade cat corral decay doubleslit2d
            dw1d efield energy h2p kepler ladder1d landau magnetic manifold
            morse1d partial pt1d qdot qpc2d rabi scene spin spins
            trapdecay tunnel tunnel1d)
        add_test(NAME Arc.${_arc}
                 COMMAND sesolver_app --selftest-${_arc})
        set_tests_properties(Arc.${_arc} PROPERTIES
            LABELS "gpu;arc"
            RESOURCE_LOCK gpu
            TIMEOUT 600)
    endforeach()
    message(STATUS "Selftest arcs registered as ctest entries (ctest -L arc).")
endif()
