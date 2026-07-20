# HARD RULE: no Qt/OpenGL/Vulkan/windowing/rendering deps -- keeps the core
# unit-testable with analytic oracles (docs/TDD_RULES.md).
# STATIC: no placeholder TU needed; each module interface unit is a real object.
add_library(sesolver_core STATIC)

# PUBLIC: app/test targets must `import` these modules across the target boundary.
target_sources(sesolver_core
    PUBLIC FILE_SET CXX_MODULES FILES
        src/marching_cubes_tables.ixx
        src/colormap.ixx
        src/vram_budget.ixx
        src/flow.ixx
        src/cross_section.ixx
        src/camera.ixx
        src/volume.ixx
        src/emission.ixx
        src/vec.ixx
        src/grid.ixx
        src/spectral.ixx
        src/potential.ixx
        src/wavepacket.ixx
        src/sphere.ixx
        src/harmonics.ixx
        src/parallel.ixx
        src/field.ixx
        src/marching_cubes.ixx
        src/fft.ixx
        src/decay.ixx
        src/measurement.ixx
        src/radial.ixx
        src/spheroidal.ixx
        src/h2plus_atlas_data.ixx
        src/observables.ixx
        src/bloch.ixx
        src/ladder.ixx
        src/lattice2d.ixx
        src/heightfield.ixx
        src/phasor.ixx
        src/spectrum1d.ixx
        src/rotation.ixx
        src/propagator.ixx
        src/imaginary_time.ixx
        src/mcwf1d.ixx
        src/spin.ixx
        src/spinlattice.ixx
        src/spinexact.ixx
        src/sampling.ixx
        src/projection.ixx
        src/drive.ixx
        src/magnetic.ixx
        src/simulation.ixx)
target_compile_features(sesolver_core PUBLIC cxx_std_20)
target_link_libraries(sesolver_core PRIVATE ses_warnings)
target_link_libraries(sesolver_core PUBLIC ses_arch)

# ses.parallel (std::jthread pool) not OpenMP: MSVC miscompiles omp in C++20
# module interfaces (silent zero-output), and fixed-chunk reductions are
# bitwise-reproducible.
find_package(Threads REQUIRED)
target_link_libraries(sesolver_core PUBLIC Threads::Threads)

# _CRT_SECURE_NO_WARNINGS: MSVC std::getenv (SES_PARALLEL_WORKERS override).
if(MSVC)
    target_compile_definitions(sesolver_core PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
