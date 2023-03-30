#include <stdio.h>
#include "backprop.h"

void create_dataset(Matrix2D features[1], Matrix2D labels[1])
{
    double min_x = -10.0, max_x = 10.0;
    int n_examples = 10001;
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

void sample(const FFModel model[1], int num_samples)
{
    double x[num_samples];
    Matrix2D features = (Matrix2D){num_samples, 1, x};
    for (int i = 0; i < num_samples; i++)
        x[i] = random_uniform_double() * 20.0 - 10.0;

    Matrix2D* pred = model_forward(model, &features);

    for (int i = 0; i < num_samples; i++)
        printf("sin(%f) = %f, pred = %f\n", x[i], sin(x[i]), pred->data[i]);
}

int main(int argc, char** argv)
{
    srand(time(NULL));

    FFModel model[1];
    create_model(model);

    Matrix2D features[1];
    Matrix2D labels[1];
    create_dataset(features, labels);

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
    sample(model, batch_size);

    free_matrix(features);
    free_matrix(labels);
    free_adam(opt_cache);
    free_model(model);

    return 0;
}
