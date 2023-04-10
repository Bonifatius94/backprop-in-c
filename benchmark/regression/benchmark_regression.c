#include <stdio.h>
#include <time.h>
#include "backprop.h"

void create_dataset(Matrix2D features[1], Matrix2D labels[1])
{
    double min_x = -10.0, max_x = 10.0;
    int n_examples = 10000;
    double step = (max_x - min_x) / n_examples;

    zeros(features, n_examples, 1);
    zeros(labels, n_examples, 1);

    for (int i = 0; i < n_examples; i++)
    {
        double x = (i - (n_examples / 2)) * step;
        double y = sin(x);
        features->data[i] = x;
        labels->data[i] = y;
    }
}

void create_model(FFModel model[1])
{
    DenseLayer* fcc_0 = (DenseLayer*)malloc(sizeof(DenseLayer));
    DenseLayer* fcc_1 = (DenseLayer*)malloc(sizeof(DenseLayer));
    DenseLayer* fcc_2 = (DenseLayer*)malloc(sizeof(DenseLayer));
    *fcc_0 = (DenseLayer){Dense, 0, 64};
    *fcc_1 = (DenseLayer){Dense, 0, 64};
    *fcc_2 = (DenseLayer){Dense, 0,  1};

    AbstractLayer* act_0 = (AbstractLayer*)malloc(sizeof(AbstractLayer));
    AbstractLayer* act_1 = (AbstractLayer*)malloc(sizeof(AbstractLayer));
    *act_0 = (AbstractLayer){ReLU, 0, 0};
    *act_1 = (AbstractLayer){ReLU, 0, 0};

    model->num_layers = 5;
    model->layers = (AbstractLayer**)malloc(sizeof(AbstractLayer*) * model->num_layers);
    model->layers[0] = (AbstractLayer*)fcc_0;
    model->layers[1] = (AbstractLayer*)act_0;
    model->layers[2] = (AbstractLayer*)fcc_1;
    model->layers[3] = (AbstractLayer*)act_1;
    model->layers[4] = (AbstractLayer*)fcc_2;
}

void train_benchmark(Matrix2D features[1], Matrix2D labels[1])
{
    FFModel model[1];
    create_model(model);

    int batch_size = 64;
    compile_model(model, features->num_cols, batch_size);

    AdamCache opt_cache[1];
    compile_adam(model, opt_cache, true);

    int num_epochs = 10;
    double train_split = 0.8;
    double learn_rate = 0.01;
    training(
        model, features, labels, num_epochs,
        batch_size, learn_rate, train_split,
        ADAM, opt_cache);

    free_adam(opt_cache);
    free_model(model);
}

int main(int argc, char** argv)
{
    srand(time(NULL));

    Matrix2D features[1];
    Matrix2D labels[1];
    create_dataset(features, labels);

    int repetitions = 10;
    clock_t start, end;
    double cpu_time_in_secs;

    start = clock();
    for (int i = 0; i < repetitions; i++)
        train_benchmark(features, labels);
    end = clock();
    cpu_time_in_secs = ((double) (end - start)) / CLOCKS_PER_SEC / repetitions;
    printf("time per training: %.2f s\n", cpu_time_in_secs);

    free_matrix(features);
    free_matrix(labels);

    return 0;
}
