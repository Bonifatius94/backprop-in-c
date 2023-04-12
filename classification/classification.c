#include <stdio.h>
#include <time.h>
#include "backprop.h"
#include "mnist.h"

void create_model(FFModel model[1])
{
    DenseLayer* fcc_0 = (DenseLayer*)malloc(sizeof(DenseLayer));
    DenseLayer* fcc_1 = (DenseLayer*)malloc(sizeof(DenseLayer));
    DenseLayer* fcc_2 = (DenseLayer*)malloc(sizeof(DenseLayer));
    *fcc_0 = (DenseLayer){Dense, 0, 400};
    *fcc_1 = (DenseLayer){Dense, 0, 400};
    *fcc_2 = (DenseLayer){Dense, 0,  10};

    AbstractLayer* act_0 = (AbstractLayer*)malloc(sizeof(AbstractLayer));
    AbstractLayer* act_1 = (AbstractLayer*)malloc(sizeof(AbstractLayer));
    AbstractLayer* act_2 = (AbstractLayer*)malloc(sizeof(AbstractLayer));
    *act_0 = (AbstractLayer){ReLU, 0, 0};
    *act_1 = (AbstractLayer){ReLU, 0, 0};
    // *act_2 = (AbstractLayer){Softmax, 0, 0};

    model->num_layers = 5;
    model->layers = (AbstractLayer**)malloc(sizeof(AbstractLayer*) * model->num_layers);
    model->layers[0] = (AbstractLayer*)fcc_0;
    model->layers[1] = (AbstractLayer*)act_0;
    model->layers[2] = (AbstractLayer*)fcc_1;
    model->layers[3] = (AbstractLayer*)act_1;
    model->layers[4] = (AbstractLayer*)fcc_2;
    // model->layers[5] = (AbstractLayer*)act_2;
}

void free_dataset(Dataset ds[1])
{
    free_matrix(ds->train_x);
    free_matrix(ds->train_y);
    free_matrix(ds->test_x);
    free_matrix(ds->test_y);
}

void log_loss(int epoch, double loss)
{
    printf("epoch %d, loss=%f\n", epoch, loss);
}

int main(int argc, char** argv)
{
    srand(time(NULL));

    FFModel model[1];
    create_model(model);

    Dataset dataset[1];
    mnist_load("../../mnist_data", (MnistDataset*)dataset);

    int batch_size = 64;
    compile_model(model, dataset->train_x->num_cols, batch_size);

    AdamCache opt_cache[1];
    compile_adam(model, opt_cache, true);

    int num_epochs = 100;
    double learn_rate = 0.01;
    training(
        model, dataset, num_epochs,
        batch_size, learn_rate,
        ADAM, opt_cache, &log_loss);

    free_dataset(dataset);
    free_adam(opt_cache);
    free_model(model);

    return 0;
}
