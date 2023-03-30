
# Backprop - Deep Learning Reference Implementation

## About
This repository is supposed to demonstrate that deep learning with low-level
languages like C isn't too hard to pull off, which is still a common belief
among artificial intelligence engineers.

There's lots of overcompilcated math calulus material out there which largely
obscures the fact that the main computation during training is just a few
very easy matrix operations which can be programmed by anyone within a day.

And not to mention, energy consumption is actually critical considering
deep learning is so widely used nowadays.
Algorithms allocating and copying way too much data
such that the process takes 10-100x more compute than necessary is really
a shame for our engineering discipline. Investing in efficient programs
is not only good for the climate but also for our AWS bills.

## Quickstart

```sh
git clone https://github.com/Bonifatius94/backprop-in-c
```

```sh
sudo apt-get update && \
    sudo apt-get install -y build-essential cmake
```

```sh
./build.sh
```

```sh
./train.sh
```
