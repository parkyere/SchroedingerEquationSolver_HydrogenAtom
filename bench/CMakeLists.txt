# Not a ctest target: timing flaky.
add_executable(sesolver_bench
    bench_main.cpp
)
target_link_libraries(sesolver_bench PRIVATE sesolver_core ses_warnings)
target_compile_features(sesolver_bench PRIVATE cxx_std_20)

# Bakes atlas -> core/src/h2plus_atlas_data.ixx; run manually.
add_executable(sesolver_genatlas
    gen_h2plus_atlas.cpp
)
target_link_libraries(sesolver_genatlas PRIVATE sesolver_core ses_warnings)
target_compile_features(sesolver_genatlas PRIVATE cxx_std_20)
