#!/bin/bash
set -e

# make sure there's an empty build folder
if [ -d "./build" ]; then rm -rf build; fi
mkdir build

# run the build and tests (release mode)
pushd build
    cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build .
    CTEST_OUTPUT_ON_FAILURE=1 ctest
popd

pushd build/benchmark
    ./benchmark_regression
popd

pushd benchmark
    python3 tf_regression.py
popd

pushd benchmark
    python3 np_regression.py
popd
