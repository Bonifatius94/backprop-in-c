add_executable(benchmark_regression benchmark_regression.c)
target_include_directories(benchmark_regression PRIVATE ../../include)
target_link_libraries(benchmark_regression m)
