#!/bin/bash
set -e

# make sure there's an empty mnist_data folder
if [ -d "./mnist_data" ]; then rm -rf mnist_data; fi
mkdir mnist_data

# download the mnist dataset files
pushd mnist_data
    wget https://github.com/apaz-cli/MNIST-dataloader-for-C/raw/master/data/train-images.idx3-ubyte
    wget https://github.com/apaz-cli/MNIST-dataloader-for-C/raw/master/data/train-labels.idx1-ubyte
    wget https://github.com/apaz-cli/MNIST-dataloader-for-C/raw/master/data/t10k-images.idx3-ubyte
    wget https://github.com/apaz-cli/MNIST-dataloader-for-C/raw/master/data/t10k-labels.idx1-ubyte
popd
