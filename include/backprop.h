#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include "matrix_ops.h"

#pragma once

/* ========================= */
/*      NEURAL NETWORK       */
/* ========================= */

typedef enum LayerType {
    Dense,
    ReLU
} LayerType;

typedef struct AbstractLayer {
    LayerType type;
    int input_dims;
    int output_dims;
} AbstractLayer;

typedef struct DenseLayer {
    LayerType type;
    int input_dims;
    int output_dims;
    Matrix2D weights[1];
    Matrix2D biases[1];
} DenseLayer;

typedef struct FFLayerCache {
    Matrix2D inputs[1];
    Matrix2D outputs[1];
    Matrix2D deltas_in[1];
    Matrix2D deltas_out[1];
    Matrix2D gradients[1];
} FFLayerCache;

typedef struct FFModel {
    int num_layers;
    AbstractLayer** layers;
    FFLayerCache* tape;
} FFModel;

void dense_unpack_grads(const DenseLayer layer[1], const Matrix2D flat_grads[1], Matrix2D grads[2])
{
    int offset = layer->input_dims * layer->output_dims;
    grads[0] = (Matrix2D){layer->input_dims, layer->output_dims, flat_grads->data};
    grads[1] = (Matrix2D){1, layer->output_dims, flat_grads->data + offset};
}

void dense_forward(const DenseLayer layer[1], FFLayerCache cache[1])
{
    matmul(cache->inputs, layer->weights, cache->outputs, MATMUL_NN);
    batch_rowadd(cache->outputs, layer->biases, cache->outputs);
}

void dense_backward(const DenseLayer layer[1], FFLayerCache cache[1])
{
    Matrix2D dense_grads[2];
    dense_unpack_grads(layer, cache->gradients, dense_grads);
    Matrix2D* weight_grads = &dense_grads[0];
    Matrix2D* bias_grads = &dense_grads[1];
    matmul(cache->inputs, cache->deltas_in, weight_grads, MATMUL_TN);
    batch_mul(weight_grads, 1.0 / cache->inputs->num_rows, weight_grads);
    batch_colmean(cache->deltas_in, bias_grads);
    if (cache->deltas_out[0].data != NULL)
        matmul(cache->deltas_in, layer->weights, cache->deltas_out, MATMUL_NT);
}

void dense_apply_grads(DenseLayer layer[1], const Matrix2D gradients[1])
{
    Matrix2D grads[2];
    dense_unpack_grads(layer, gradients, grads);
    Matrix2D* weight_grads = &grads[0];
    Matrix2D* bias_grads = &grads[1];
    elemdiff(layer->weights, weight_grads, layer->weights);
    elemdiff(layer->biases, bias_grads, layer->biases);
}

void relu_forward(FFLayerCache cache[1])
{
    batch_max(cache->inputs, 0.0, cache->outputs);
}

void relu_backward(FFLayerCache cache[1])
{
    batch_geq(cache->inputs, 0.0, cache->deltas_out);
    elemmul(cache->deltas_in, cache->deltas_out, cache->deltas_out);
}

void layer_forward(const AbstractLayer layer[1], FFLayerCache cache[1])
{
    if (layer->type == Dense)
        dense_forward((DenseLayer*)layer, cache);
    else if (layer->type == ReLU)
        relu_forward(cache);
}

void layer_backward(const AbstractLayer layer[1], FFLayerCache cache[1])
{
    if (layer->type == Dense)
        dense_backward((DenseLayer*)layer, cache);
    else if (layer->type == ReLU)
        relu_backward(cache);
}

void layer_apply_grads(AbstractLayer layer[1], const FFLayerCache cache[1])
{
    if (layer->type == Dense)
        dense_apply_grads((DenseLayer*)layer, cache->gradients);
}

Matrix2D* model_forward(const FFModel model[1], const Matrix2D features[1])
{
    model->tape[0].inputs[0] = *features;
    for (int i = 0; i < model->num_layers; i++)
        layer_forward(model->layers[i], model->tape + i);
    return model->tape[model->num_layers - 1].outputs;
}

void model_backward(const FFModel model[1], const Matrix2D loss_deltas[1])
{
    model->tape[model->num_layers - 1].deltas_in[0] = *loss_deltas;
    for (int i = model->num_layers - 1; i >= 0; i--)
        layer_backward(model->layers[i], model->tape + i);
}

void model_apply_grads(FFModel model[1])
{
    for (int i = 0; i < model->num_layers; i++)
        layer_apply_grads(model->layers[i], &model->tape[i]);
}

/* ========================= */
/*          FF MODEL         */
/* ========================= */

void compile_model(FFModel model[1], int feature_dims, int batch_size)
{
    Matrix2D pred_caches[model->num_layers];
    Matrix2D delta_caches[model->num_layers];
    int input_dims = feature_dims;

    for (int i = 0; i < model->num_layers; i++)
    {
        AbstractLayer* abs_layer = model->layers[i];
        if (abs_layer->type == Dense)
        {
            DenseLayer* layer = (DenseLayer*)abs_layer;
            layer->input_dims = input_dims;
            zeros_with_cache(layer->weights, layer->input_dims, layer->output_dims);
            zeros_with_cache(layer->biases, 1, layer->output_dims);
            random_normal(layer->weights, 0.0, 0.1);
            input_dims = layer->output_dims;
        }
        else if (abs_layer->type == ReLU)
        {
            abs_layer->input_dims = input_dims;
            abs_layer->output_dims = input_dims;
        }

        zeros_with_cache(&pred_caches[i], batch_size, abs_layer->output_dims);
        zeros_with_cache(&delta_caches[i], batch_size, abs_layer->output_dims);
    }

    model->tape = (FFLayerCache*)malloc(sizeof(FFLayerCache) * model->num_layers);
    for (int i = 0; i < model->num_layers; i++)
    {
        Matrix2D grads = EMPTY_MATRIX;
        AbstractLayer* abs_layer = model->layers[i];
        if (abs_layer->type == Dense)
        {
            DenseLayer* layer = (DenseLayer*)abs_layer;
            size_t num_params = layer->weights->num_rows * layer->weights->num_cols
                + layer->biases->num_cols;
            zeros_with_cache(&grads, 1, num_params);
        }

        /* info: input features and loss deltas are external matrices */
        FFLayerCache* cache = &model->tape[i];
        cache->inputs[0] = (i > 0) ? pred_caches[i - 1] : EMPTY_MATRIX;
        cache->outputs[0] = pred_caches[i];
        cache->deltas_in[0] = delta_caches[i];
        cache->deltas_out[0] = (i > 0) ? delta_caches[i - 1] : EMPTY_MATRIX;
        cache->gradients[0] = grads;
    }
}

void free_model(FFModel* model)
{
    for (int i = 0; i < model->num_layers; i++)
    {
        AbstractLayer* abs_layer = model->layers[i];
        if (abs_layer->type == Dense)
        {
            DenseLayer* layer = (DenseLayer*)abs_layer;
            free_matrix(layer->weights);
            free_matrix(layer->biases);
            free(layer);
        }
        else if (abs_layer->type == ReLU)
        {
            free(abs_layer);
        }

        FFLayerCache layer_cache = model->tape[i];
        free_matrix(layer_cache.outputs);
        free_matrix(layer_cache.deltas_in);
        free_matrix(layer_cache.gradients);
    }

    free(model->layers);
    free(model->tape);
}

/* ========================= */
/*         OPTIMIZER         */
/* ========================= */

typedef enum OptimizerType {
    NAIVE_SGD, ADAM
} OptimizerType;

typedef struct AdamCache {
    int num_layers;
    double beta_1;
    double beta_2;
    double epsilon;
    Matrix2D* m;
    Matrix2D* v;
    Matrix2D* m_temp;
    Matrix2D* v_temp;
    double beta_1t;
    double beta_2t;
} AdamCache;

#define DEFAULT_ADAM_CONFIG ((AdamCache){0, 0.9, 0.999, 1e-8, NULL, NULL, NULL, NULL, 1.0, 1.0})

void compile_adam(const FFModel model[1], AdamCache cache[1], bool use_defaults)
{
    if (use_defaults)
        *cache = DEFAULT_ADAM_CONFIG;
    cache->num_layers = model->num_layers;
    cache->m = (Matrix2D*)malloc(model->num_layers * sizeof(Matrix2D));
    cache->v = (Matrix2D*)malloc(model->num_layers * sizeof(Matrix2D));
    cache->m_temp = (Matrix2D*)malloc(model->num_layers * sizeof(Matrix2D));
    cache->v_temp = (Matrix2D*)malloc(model->num_layers * sizeof(Matrix2D));

    for (int i = 0; i < model->num_layers; i++)
    {
        zeros_like(model->tape[i].gradients, &cache->m[i]);
        zeros_like(model->tape[i].gradients, &cache->v[i]);
        zeros_like(model->tape[i].gradients, &cache->m_temp[i]);
        zeros_like(model->tape[i].gradients, &cache->v_temp[i]);
    }
}

void free_adam(AdamCache cache[1])
{
    free(cache->m);
    free(cache->v);
    free(cache->m_temp);
    free(cache->v_temp);
}

void adjust_grads_naive_sgd(FFModel model[1], double learn_rate)
{
    for (int i = 0; i < model->num_layers; i++)
    {
        Matrix2D* grads = model->tape[i].gradients;
        if (grads->data == NULL) continue;
        batch_mul(grads, learn_rate, grads);
    }
}

void adjust_grads_adam(FFModel model[1], AdamCache cache[1], double learn_rate)
{
    for (int i = 0; i < model->num_layers; i++)
    {
        Matrix2D* grads = model->tape[i].gradients;
        Matrix2D* m = &cache->m[i];
        Matrix2D* v = &cache->v[i];
        Matrix2D* m_temp = &cache->m_temp[i];
        Matrix2D* v_temp = &cache->v_temp[i];
        if (grads->data == NULL) continue;

        batch_mul(m, cache->beta_1, m);
        batch_mul(grads, 1 - cache->beta_1, m_temp);
        elemsum(m, m_temp, m);

        batch_mul(v, cache->beta_2, v);
        elemmul(grads, grads, v_temp);
        batch_mul(v_temp, 1 - cache->beta_2, v_temp);
        elemsum(v, v_temp, v);

        cache->beta_1t *= cache->beta_1;
        cache->beta_2t *= cache->beta_2;

        batch_div(m, 1 - cache->beta_1t, m_temp);
        batch_div(v, 1 - cache->beta_2t, v_temp);

        batch_mul(m_temp, learn_rate, m_temp);
        batch_sqrt(v_temp, v_temp);
        batch_sum(v_temp, cache->epsilon, v_temp);
        elemdiv(m_temp, v_temp, grads);
    }
}

void optimize(FFModel model[1], OptimizerType opt_type, void* cache, double learn_rate)
{
    if (opt_type == ADAM)
        adjust_grads_adam(model, (AdamCache*)cache, learn_rate);
    else if (opt_type == NAIVE_SGD)
        adjust_grads_naive_sgd(model, learn_rate);
    model_apply_grads(model);
}

/* ========================= */
/*         TRAINING          */
/* ========================= */

typedef struct Dataset {
    Matrix2D train_x[1];
    Matrix2D train_y[1];
    Matrix2D test_x[1];
    Matrix2D test_y[1];
} Dataset;

typedef void (*loss_logger)(int, double);

double mse_loss(const Matrix2D y_pred[1], const Matrix2D y_true[1])
{
    assert(y_true->num_rows == y_pred->num_rows);
    assert(y_true->num_cols == y_pred->num_cols);

    double loss = 0.0;
    for (int row = 0; row < y_true->num_rows; row++)
    {
        for (int col = 0; col < y_true->num_cols; col++)
        {
            int i = row * y_true->num_cols + col;
            double diff = y_pred->data[i] - y_true->data[i];
            loss += diff * diff;
        }
    }
    return loss / y_true->num_rows;
}

void mse_loss_delta(const Matrix2D y_pred[1], const Matrix2D y_true[1], Matrix2D delta[1])
{
    elemdiff(y_pred, y_true, delta);
}

void training_step(
    FFModel model[1], const Matrix2D x[1], const Matrix2D y_true[1],
    OptimizerType opt_type, void* cache, double learn_rate)
{
    Matrix2D* y_pred = model_forward(model, x);
    Matrix2D* deltas = (&model->tape[model->num_layers - 1])->deltas_in;
    mse_loss_delta(y_pred, y_true, deltas);
    model_backward(model, deltas);
    optimize(model, opt_type, cache, learn_rate);
}

void shuffle_dataset(Matrix2D x[1], Matrix2D y[1])
{
    assert(x->num_rows == y->num_rows);

    int perm[x->num_rows];
    for (int i = 0; i < x->num_rows; i++)
        perm[i] = i;

    permutate(perm, x->num_rows);
    shuffle_rows(x, perm);
    shuffle_rows(y, perm);
}

void training(
    FFModel model[1], Dataset ds[1],
    int num_epochs, int batch_size, double learn_rate,
    OptimizerType opt_type, void* opt_cache, loss_logger logger)
{
    assert(ds->train_x->num_rows == ds->train_y->num_rows);
    assert(ds->test_x->num_rows == ds->test_y->num_rows);

    int num_train_examples = ds->train_x->num_rows;
    int num_test_examples = ds->test_x->num_rows;
    int input_dims = ds->train_x->num_cols;
    int output_dims = ds->train_y->num_cols;
    int num_train_batches = num_train_examples / batch_size;
    int num_test_batches = num_test_examples / batch_size;

    Matrix2D features_cache[1], labels_cache[1];
    zeros(features_cache, batch_size, input_dims);
    zeros(labels_cache, batch_size, output_dims);
    Matrix2D batch_x = (Matrix2D){batch_size, input_dims, ds->train_x->data, features_cache->data};
    Matrix2D batch_y = (Matrix2D){batch_size, output_dims, ds->train_y->data, labels_cache->data};

    int train_perm[num_train_batches];
    for (int i = 0; i < num_train_batches; i++)
        train_perm[i] = i;

    shuffle_dataset(ds->train_x, ds->train_y);

    for (int epoch = 0; epoch < num_epochs; epoch++)
    {
        permutate(train_perm, num_train_batches);

        for (int i = 0; i < num_train_batches; i++)
        {
            batch_x.data = ds->train_x->data + train_perm[i] * input_dims;
            batch_y.data = ds->train_y->data + train_perm[i] * output_dims;
            training_step(model, &batch_x, &batch_y, opt_type, opt_cache, learn_rate);
        }

        double loss = 0.0;
        batch_x.data = ds->test_x->data;
        batch_y.data = ds->test_y->data;
        for (int i = 0; i < num_test_batches; i++)
        {
            Matrix2D* pred_y = model_forward(model, &batch_x);
            loss += mse_loss(&batch_y, pred_y);
            batch_x.data += batch_size * input_dims;
            batch_y.data += batch_size * output_dims;
        }
        logger(epoch, loss / num_test_batches);
    }

    free_matrix(features_cache);
    free_matrix(labels_cache);
}
