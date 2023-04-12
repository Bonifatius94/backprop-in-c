#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <assert.h>
#include "matrix_ops.h"

#pragma once

typedef struct MnistDataset {
    Matrix2D train_x[1];
    Matrix2D train_y[1];
    Matrix2D test_x[1];
    Matrix2D test_y[1];
} MnistDataset;

uint32_t revert_endianess_u32(uint32_t i)
{
    return ((i & 0x000000FF) << 24) | ((i & 0x0000FF00) <<  8)
         | ((i & 0x00FF0000) >>  8) | ((i & 0xFF000000) >> 24);
}

void read_u32_reverted_endianess(int fd, uint32_t* i)
{
    ssize_t ret = read(fd, i, 4);
    assert(ret != -1);
    *i = revert_endianess_u32(*i);
}

void normalize_grayscale_pixels(uint8_t* pixels, double* features, size_t len)
{
    for (size_t i = 0; i < len; i++)
        features[i] = (((double)pixels[i]) / 127.5) - 1.0;
}

void read_mnist_images(const char* file_path, Matrix2D images[1])
{
    int fd = open(file_path, O_RDONLY);
    assert(fd != -1);

    uint32_t magic_number, img_height, img_width, num_imgs;
    read_u32_reverted_endianess(fd, &magic_number);
    read_u32_reverted_endianess(fd, &num_imgs);
    read_u32_reverted_endianess(fd, &img_height);
    read_u32_reverted_endianess(fd, &img_width);
    assert(magic_number == 2051);

    size_t max_bytes = num_imgs * img_height * img_width;
    zeros(images, num_imgs, img_height * img_width);

    size_t buf_len = img_height * img_width;
    uint8_t img_buf[buf_len];
    for (size_t i = 0; i < num_imgs; i++)
    {
        ssize_t ret = read(fd, img_buf, buf_len);
        assert(ret != -1);
        normalize_grayscale_pixels(img_buf, images->data + i, buf_len);
    }

    fd = close(fd);
    assert(fd != -1);
}

void read_mnist_labels(const char* file_path, Matrix2D labels_onehot[1])
{
    int fd = open(file_path, O_RDONLY);
    assert(fd != -1);

    uint32_t magic_number, num_imgs;
    read_u32_reverted_endianess(fd, &magic_number);
    read_u32_reverted_endianess(fd, &num_imgs);
    assert(magic_number == 2049);

    const uint32_t num_classes = 10;
    zeros(labels_onehot, num_imgs, num_classes);

    uint8_t label_buf[num_imgs];
    ssize_t ret = read(fd, label_buf, num_imgs);
    assert(ret != -1);

    for (size_t i = 0; i < num_imgs; i++)
        labels_onehot->data[i * num_classes + label_buf[i]] = 1.0;

    fd = close(fd);
    assert(fd != -1);
}

void path_join(char* path_buf, const char* path_parent, const char* path_child)
{
    size_t i = strlen(path_parent);
    assert(i < 200);
    strcpy(path_buf, path_parent);
    if (path_buf[i - 1] != '/')
        path_buf[i++] = '/';
    path_buf[i] = '\0';
    strcat(path_buf, path_child);
}

void mnist_load(const char* data_folder, MnistDataset mnist[1])
{
    const char file_trainx[] = "train-images.idx3-ubyte";
    const char file_trainy[] = "train-labels.idx1-ubyte";
    const char file_testx[] = "t10k-images.idx3-ubyte";
    const char file_testy[] = "t10k-labels.idx1-ubyte";

    char path_trainx[256], path_trainy[256];
    char path_testx[256], path_testy[256];
    path_join(path_trainx, data_folder, file_trainx);
    path_join(path_trainy, data_folder, file_trainy);
    path_join(path_testx, data_folder, file_testx);
    path_join(path_testy, data_folder, file_testy);

    read_mnist_images(path_trainx, mnist->train_x);
    read_mnist_labels(path_trainy, mnist->train_y);
    read_mnist_images(path_testx, mnist->test_x);
    read_mnist_labels(path_testy, mnist->test_y);
}
