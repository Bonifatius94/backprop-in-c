#include <ctype.h>
#include <stddef.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>

/* ========================= */
/*     RANDOM OPERATIONS     */
/* ========================= */

/* generate a uniform random double between [0, 1]. */
double random_uniform_double()
{
    return (double)rand() / RAND_MAX;
}

/* generate a uniform random integer between [0, n-1]. */
int random_int(int n)
{
    return ((int)(random_uniform_double() * n)) % n;
}

/* Generate a random normal distributed floating-point value. */
double random_normal_double(double center, double std_dev)
{
    const double epsilon = 1.19e-07;
    const double two_pi = 2.0 * M_PI;

    double u1, u2;
    do { u1 = random_uniform_double(); }
    while (u1 <= epsilon);
    u2 = random_uniform_double();

    double mag = std_dev * sqrt(-2.0 * log(u1));
    if (random_uniform_double() > 0.5)
        return mag * cos(two_pi * u2) + center;
    else
        return mag * sin(two_pi * u2) + center;
}

void permutate(int perm[], int len)
{
    for (int i = len - 1; i > 0; i--)
    {
        int j = random_int(i + 1);

        if (i != j)
        {
            int temp = perm[i];
            perm[i] = perm[j];
            perm[j] = temp;
        }
    }
}

/* ========================= */
/*     MATRIX OPERATIONS     */
/* ========================= */

typedef struct Matrix2D {
    int num_rows;
    int num_cols;
    double* data;
    double* cache;
} Matrix2D;

#define EMPTY_MATRIX ((Matrix2D){0, 0, NULL, NULL})

typedef enum MatmulFlags {
    MATMUL_NN, MATMUL_TN, MATMUL_NT, MATMUL_TT
} MatmulFlags;

/* Initialize an unallocated matrix with zeros according to the shape. */
void zeros(Matrix2D a1[1], int num_rows, int num_cols)
{
    a1->num_rows = num_rows;
    a1->num_cols = num_cols;
    int size = (int)(ceil((num_rows * num_cols) / 16.0)) * 16;
    a1->data = (double*)calloc(size, sizeof(double));
    a1->cache = NULL;
}

/* Initialize an unallocated matrix with zeros according to the shape.
   Moreover, initialize an equally sized cache for faster matrix ops. */
void zeros_with_cache(Matrix2D a1[1], int num_rows, int num_cols)
{
    a1->num_rows = num_rows;
    a1->num_cols = num_cols;
    int size = (int)(ceil((num_rows * num_cols) / 16.0)) * 16;
    a1->data = (double*)calloc(size, sizeof(double));
    a1->cache = (double*)calloc(size, sizeof(double));
    /* info: allocate twice as much to make use of transpose caching in matmul(),
             align to 16-byte memory layout to support 512-bit AVX2 ops */
}

/* Initialize an unallocated matrix with zeros according to the original's shape. */
void zeros_like(const Matrix2D a1[1], Matrix2D res[1])
{
    res->num_rows = a1->num_rows;
    res->num_cols = a1->num_cols;
    res->data = NULL;

    int size = (int)(ceil((a1->num_rows * a1->num_cols) / 16.0)) * 16;
    if (a1->data != NULL)
        res->data = (double*)calloc(size, sizeof(double));
    if (a1->cache != NULL)
        res->cache = (double*)calloc(size, sizeof(double));
}

/* Initialize a pre-allocated matrix with the same data as the original. */
void copy(const Matrix2D orig[1], Matrix2D copy[1])
{
    for (int i = 0; i < copy->num_rows * copy->num_cols; i++)
        copy->data[i] = orig->data[i];
}

/* Initialize a pre-allocated matrix with random normal distributed values. */
void random_normal(Matrix2D res[1], double center, double std_dev)
{
    for (int i = 0; i < res->num_rows * res->num_cols; i++)
        res->data[i] = random_normal_double(center, std_dev);
}

double dot_product(Matrix2D v1[1], Matrix2D v2[1])
{
    assert(v1->num_rows == 1 && v2->num_rows == 1);
    assert(v1->num_cols == v2->num_cols);

    /* TODO: use SIMD to speed this up */
    double sum = 0.0;
    for (int i = 0; i < v1->num_cols; i++)
        sum += v1->data[i] * v2->data[i];
    return sum;
}

void transpose(const Matrix2D a1[1], Matrix2D res[1])
{
    assert(a1->num_rows == res->num_cols);
    assert(a1->num_cols == res->num_rows);
    for (int i = 0; i < a1->num_rows; i++)
        for (int j = 0; j < a1->num_cols; j++)
            res->data[j * a1->num_rows + i] = a1->data[i * a1->num_cols + j];
    /* TODO: think of implementing a faster transpose */
}

/* Multiply two matrices, write the result in a third matrix.
   Input matrices are assumed as transposed according to the flags. */
void matmul(const Matrix2D a1[1], const Matrix2D a2[1], Matrix2D res[1], MatmulFlags flags)
{
    bool a1_normal = flags == MATMUL_NN || flags == MATMUL_NT;
    bool a2_normal = flags == MATMUL_NN || flags == MATMUL_TN;
    int l = a1_normal ? a1->num_rows : a1->num_cols;
    int m = a1_normal ? a1->num_cols : a1->num_rows;
    int m2 = a2_normal ? a2->num_rows : a2->num_cols;
    int n = a2_normal ? a2->num_cols : a2->num_rows;
    assert(res->num_rows == l && res->num_cols == n && m == m2);
    /* matrix shapes: (l, m) x (m, n) = (l, n) */

    /* transpose matrices into row layout */
    Matrix2D a1_rl = (Matrix2D){l, m, a1->cache, NULL};
    Matrix2D a2_rl = (Matrix2D){n, m, a2->cache, NULL};
    if (!a1_normal)
        transpose(a1, &a1_rl);
    else
        a1_rl.data = a1->data;
    if (a2_normal)
        transpose(a2, &a2_rl);
    else
        a2_rl.data = a2->data;
    a1_rl.num_rows = 1;
    a2_rl.num_rows = 1;

    for (int i = 0; i < l; i++)
    {
        double* a2_rl_data = a2_rl.data;

        for (int j = 0; j < n; j++)
        {
            double prod = dot_product(&a1_rl, &a2_rl);
            res->data[i * n + j] = prod;
            a2_rl.data += m;
        }

        a1_rl.data += m;
        a2_rl.data = a2_rl_data;
    }
}

void elemsum(const Matrix2D a1[1], const Matrix2D a2[1], Matrix2D res[1])
{
    assert(a1->num_rows == a2->num_rows);
    assert(a1->num_cols == a2->num_cols);
    assert(a1->num_rows == res->num_rows);
    assert(a1->num_cols == res->num_cols);

    /* TODO: use SIMD to speed this up */
    for (int i = 0; i < a1->num_rows * a1->num_cols; i++)
        res->data[i] = a1->data[i] + a2->data[i];
}

void elemdiff(const Matrix2D a1[1], const Matrix2D a2[1], Matrix2D res[1])
{
    assert(a1->num_rows == a2->num_rows);
    assert(a1->num_cols == a2->num_cols);
    assert(a1->num_rows == res->num_rows);
    assert(a1->num_cols == res->num_cols);

    /* TODO: use SIMD to speed this up */
    for (int i = 0; i < a1->num_rows * a1->num_cols; i++)
        res->data[i] = a1->data[i] - a2->data[i];
}

void elemmul(const Matrix2D a1[1], const Matrix2D a2[1], Matrix2D res[1])
{
    assert(a1->num_rows == a2->num_rows);
    assert(a1->num_cols == a2->num_cols);
    assert(a1->num_rows == res->num_rows);
    assert(a1->num_cols == res->num_cols);

    /* TODO: use SIMD to speed this up */
    for (int i = 0; i < a1->num_rows * a1->num_cols; i++)
        res->data[i] = a1->data[i] * a2->data[i];
}

void elemdiv(const Matrix2D a1[1], const Matrix2D a2[1], Matrix2D res[1])
{
    assert(a1->num_rows == a2->num_rows);
    assert(a1->num_cols == a2->num_cols);
    assert(a1->num_rows == res->num_rows);
    assert(a1->num_cols == res->num_cols);

    /* TODO: use SIMD to speed this up */
    for (int i = 0; i < a1->num_rows * a1->num_cols; i++)
        res->data[i] = a1->data[i] / a2->data[i];
}

void batch_rowadd(const Matrix2D a1[1], const Matrix2D row_vec[1], Matrix2D res[1])
{
    assert(a1->num_cols == row_vec->num_cols);
    assert(a1->num_rows == res->num_rows);
    assert(a1->num_cols == res->num_cols);

    /* TODO: use SIMD to speed this up */
    int i = 0;
    for (int row = 0; row < a1->num_rows; row++)
    {
        for (int col = 0; col < a1->num_cols; col++)
        {
            res->data[i] = a1->data[i] + row_vec->data[col];
            i++;
        }
    }
}

void batch_colmean(const Matrix2D a1[1], Matrix2D res[1])
{
    assert(a1->num_cols == res->num_cols);
    assert(res->num_rows == 1);

    /* TODO: use SIMD to speed this up */
    for (int col = 0; col < a1->num_cols; col++)
    {
        double colsum = 0.0;
        for (int row = 0; row < a1->num_rows; row++)
            colsum += a1->data[row * a1->num_cols + col];
        res->data[col] = colsum / a1->num_rows;
    }
}

void batch_sum(const Matrix2D a1[1], double summand, Matrix2D res[1])
{
    assert(a1->num_rows == res->num_rows);
    assert(a1->num_cols == res->num_cols);

    /* TODO: use SIMD to speed this up */
    for (int i = 0; i < a1->num_rows * a1->num_cols; i++)
        res->data[i] = a1->data[i] + summand;
}

void batch_diff(const Matrix2D a1[1], double diff, Matrix2D res[1])
{
    batch_sum(a1, -diff, res);
}

void batch_mul(const Matrix2D a1[1], double factor, Matrix2D res[1])
{
    assert(a1->num_rows == res->num_rows);
    assert(a1->num_cols == res->num_cols);

    /* TODO: use SIMD to speed this up */
    for (int i = 0; i < a1->num_rows * a1->num_cols; i++)
        res->data[i] = a1->data[i] * factor;
}

void batch_div(const Matrix2D a1[1], double factor, Matrix2D res[1])
{
    assert(factor != 0.0);
    factor = 1 / factor;
    batch_mul(a1, factor, res);
}

void batch_sqrt(const Matrix2D a1[1], Matrix2D res[1])
{
    assert(a1->num_rows == res->num_rows);
    assert(a1->num_cols == res->num_cols);

    /* TODO: use SIMD to speed this up */
    for (int i = 0; i < a1->num_rows * a1->num_cols; i++)
        res->data[i] = sqrt(a1->data[i]);
}

void batch_max(const Matrix2D a1[1], double min, Matrix2D res[1])
{
    assert(a1->num_rows == res->num_rows);
    assert(a1->num_cols == res->num_cols);

    /* TODO: use SIMD to speed this up */
    for (int i = 0; i < a1->num_rows * a1->num_cols; i++)
        res->data[i] = a1->data[i] >= min ? a1->data[i] : min;
}

void batch_geq(const Matrix2D a1[1], double min, Matrix2D res[1])
{
    assert(a1->num_rows == res->num_rows);
    assert(a1->num_cols == res->num_cols);

    for (int row = 0; row < a1->num_rows; row++)
    {
        for (int col = 0; col < a1->num_cols; col++)
        {
            int i = row * a1->num_cols + col;
            res->data[i] = a1->data[i] >= min ? 1.0 : 0.0;
        }
    }
}

void shuffle_rows(Matrix2D a1[1], const int perm[])
{
    double temp_data[a1->num_rows * a1->num_cols];

    for (int i = 0; i < a1->num_rows; i++)
        for (int j = 0; j < a1->num_cols; j++)
            temp_data[i * a1->num_cols + j] = a1->data[perm[i] * a1->num_cols + j];

    for (int i = 0; i < a1->num_rows; i++)
        for (int j = 0; j < a1->num_cols; j++)
            a1->data[i * a1->num_cols + j] = temp_data[i * a1->num_cols + j];
}

void free_matrix(Matrix2D a1[1])
{
    if (a1->data != NULL)
        free(a1->data);
    if (a1->cache != NULL)
        free(a1->cache);
}

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
    FFModel model[1], Matrix2D features[1], Matrix2D labels[1],
    int num_epochs, int batch_size, double learn_rate, double train_split,
    OptimizerType opt_type, void* opt_cache)
{
    int num_train_examples = (int)(features->num_rows * train_split);
    int num_train_batches = num_train_examples / batch_size;
    int num_test_batches = (features->num_rows - num_train_examples) / batch_size;

    Matrix2D features_cache[1], labels_cache[1];
    zeros(features_cache, batch_size, features->num_cols);
    zeros(labels_cache, batch_size, labels->num_cols);
    Matrix2D batch_x = (Matrix2D){batch_size, features->num_cols, features->data, features_cache->data};
    Matrix2D batch_y = (Matrix2D){batch_size, labels->num_cols, labels->data, labels_cache->data};

    int train_perm[num_train_batches];
    for (int i = 0; i < num_train_batches; i++)
        train_perm[i] = i;

    shuffle_dataset(features, labels);

    for (int epoch = 0; epoch < num_epochs; epoch++)
    {
        permutate(train_perm, num_train_batches);

        for (int i = 0; i < num_train_batches; i++)
        {
            batch_x.data = features->data + train_perm[i] * features->num_cols;
            batch_y.data = labels->data + train_perm[i] * labels->num_cols;
            training_step(model, &batch_x, &batch_y, opt_type, opt_cache, learn_rate);
        }

        double loss = 0.0;
        batch_x.data = features->data + num_train_batches * features->num_cols;
        batch_y.data = labels->data + num_train_batches * labels->num_cols;
        for (int i = 0; i < num_test_batches; i++)
        {
            Matrix2D* pred_y = model_forward(model, &batch_x);
            loss += mse_loss(&batch_y, pred_y);
            batch_x.data += batch_size * features->num_cols;
            batch_y.data += batch_size * labels->num_cols;
        }
        printf("epoch %d, loss=%f\n", epoch, loss / num_test_batches);
    }

    free_matrix(features_cache);
    free_matrix(labels_cache);
}
