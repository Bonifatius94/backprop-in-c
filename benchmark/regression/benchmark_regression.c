#include <stdio.h>
#include <time.h>
#include "backprop.h"

void create_dataset(Dataset ds[1])
{
    double min_x = -10.0, max_x = 10.0;
    int n_examples = 10000;
    double train_split = 0.8;

    double x_range = max_x - min_x;
    int num_train_examples = (int)(n_examples * train_split);
    int num_test_examples = n_examples - num_train_examples;

    zeros(ds->train_x, num_train_examples, 1);
    zeros(ds->train_y, num_train_examples, 1);
    zeros(ds->test_x, num_test_examples, 1);
    zeros(ds->test_y, num_test_examples, 1);

    for (int i = 0; i < num_train_examples; i++)
    {
        double x = random_uniform_double() * x_range + min_x;
        double y = sin(x);
        ds->train_x->data[i] = x;
        ds->train_y->data[i] = y;
    }

    for (int i = 0; i < num_test_examples; i++)
    {
        double x = random_uniform_double() * x_range + min_x;
        double y = sin(x);
        ds->test_x->data[i] = x;
        ds->test_y->data[i] = y;
    }
}

void free_dataset(Dataset ds[1])
{
    free_matrix(ds->train_x);
    free_matrix(ds->train_y);
    free_matrix(ds->test_x);
    free_matrix(ds->test_y);
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

void log_loss(int epoch, double loss)
{
    printf("epoch %d, loss=%f\n", epoch, loss);
}

void train_benchmark(Dataset dataset[1])
{
    FFModel model[1];
    create_model(model);

    int batch_size = 64;
    compile_model(model, dataset->train_x->num_cols, batch_size);

    AdamCache opt_cache[1];
    compile_adam(model, opt_cache, true);

    int num_epochs = 10;
    double learn_rate = 0.01;
    training(
        model, dataset, num_epochs,
        batch_size, learn_rate,
        ADAM, opt_cache, &log_loss);

    free_adam(opt_cache);
    free_model(model);
}

int main(int argc, char** argv)
{
    srand(time(NULL));

    Dataset dataset[1];
    create_dataset(dataset);

    int repetitions = 10;
    clock_t start, end;
    double cpu_time_in_secs;

    start = clock();
    for (int i = 0; i < repetitions; i++)
        train_benchmark(dataset);
    end = clock();
    cpu_time_in_secs = ((double) (end - start)) / CLOCKS_PER_SEC / repetitions;
    printf("time per training: %.2f s\n", cpu_time_in_secs);

    free_dataset(dataset);

    return 0;
}
