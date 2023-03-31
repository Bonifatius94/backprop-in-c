cmake_minimum_required(VERSION 3.9.4)
project(backprop LANGUAGES C)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# set(CMAKE_C_FLAGS "-Wall -Wextra")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O3")

add_subdirectory(test)
enable_testing()
add_test(NAME MatrixTests COMMAND MatrixTests)
add_test(NAME ModelTests COMMAND ModelTests)

add_subdirectory(regression)
add_subdirectory(benchmark)
