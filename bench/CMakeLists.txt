# Manual micro-benchmark target (not registered with ctest; timing assertions
# are flaky by nature -- correctness lives in tests/).
add_executable(sesolver_bench
    bench_main.cpp
)
target_link_libraries(sesolver_bench PRIVATE sesolver_core ses_warnings)
target_compile_features(sesolver_bench PRIVATE cxx_std_20)
