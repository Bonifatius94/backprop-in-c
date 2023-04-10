#!/bin/bash
set -e

# make sure there's an empty build folder
if [ -d "./build" ]; then rm -rf build; fi
mkdir build

# run the build and tests (release mode)
pushd build > /dev/null
    cmake -DCMAKE_BUILD_TYPE=Release .. > /dev/null
    cmake --build . > /dev/null
    CTEST_OUTPUT_ON_FAILURE=1 ctest > /dev/null
popd > /dev/null

echo "====================================="
echo "              BENCHMARK              "
echo "====================================="

pushd build/benchmark/regression > /dev/null
    echo "reference implementation 'backprop'"
    ./benchmark_regression | tail -n 1
popd > /dev/null

echo "====================================="

pushd benchmark/regression > /dev/null
    echo "tensorflow implementation"
    python3 tf_regression.py | tail -n 1
popd > /dev/null

echo "====================================="

pushd benchmark/regression > /dev/null
    echo "numpy implementation"
    python3 np_regression.py | tail -n 1
popd > /dev/null

echo "====================================="
