#include <assert.h>
#include "backprop.h"

void create_model(FFModel model[1])
{
    DenseLayer* fcc_0 = (DenseLayer*)malloc(sizeof(DenseLayer));
    DenseLayer* fcc_1 = (DenseLayer*)malloc(sizeof(DenseLayer));
    DenseLayer* fcc_2 = (DenseLayer*)malloc(sizeof(DenseLayer));
    *fcc_0 = (DenseLayer){Dense, 0, 62};
    *fcc_1 = (DenseLayer){Dense, 0, 66};
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

void assert_dense_layer(DenseLayer layer[1], int input_dims, int output_dims)
{
    assert(layer->weights->num_rows == input_dims);
    assert(layer->weights->num_cols == output_dims);
    assert(layer->weights->data != NULL);
    assert(layer->biases->num_rows == 1);
    assert(layer->biases->num_cols == output_dims);
    assert(layer->biases->data != NULL);
}

void assert_ff_layer_cache(
    FFLayerCache cache[1], int input_dims, int output_dims,
    int batch_size, bool is_first, bool is_dense_layer)
{
    if (is_first)
    {
        assert(cache->inputs->num_rows == 0);
        assert(cache->inputs->num_cols == 0);
        assert(cache->inputs->data == NULL);

        assert(cache->deltas_out->num_rows == 0);
        assert(cache->deltas_out->num_cols == 0);
        assert(cache->deltas_out->data == NULL);
    }
    else
    {
        assert(cache->inputs->num_rows == batch_size);
        assert(cache->inputs->num_cols == input_dims);
        assert(cache->inputs->data != NULL);

        assert(cache->deltas_out->num_rows == batch_size);
        assert(cache->deltas_out->num_cols == input_dims);
        assert(cache->deltas_out->data != NULL);
    }

    assert(cache->outputs->num_rows == batch_size);
    assert(cache->outputs->num_cols == output_dims);
    assert(cache->outputs->data != NULL);

    assert(cache->deltas_in->num_rows == batch_size);
    assert(cache->deltas_in->num_cols == output_dims);
    assert(cache->deltas_in->data != NULL);

    if (is_dense_layer)
    {
        int exp_num_grads = input_dims * output_dims + output_dims;
        assert(cache->gradients->num_rows == 1);
        assert(cache->gradients->num_cols == exp_num_grads);
        assert(cache->gradients->data != NULL);
    }
    else
    {
        assert(cache->gradients->num_rows == 0);
        assert(cache->gradients->num_cols == 0);
        assert(cache->gradients->data == NULL);
    }
}

void test_create_and_free_model()
{
    FFModel model[1]; int batch_size = 32, num_features = 10;
    create_model(model);
    compile_model(model, num_features, batch_size);

    assert(model->num_layers == 5);
    assert(model->layers[0]->type == Dense);
    assert(model->layers[1]->type == ReLU);
    assert(model->layers[2]->type == Dense);
    assert(model->layers[3]->type == ReLU);
    assert(model->layers[4]->type == Dense);
    assert_dense_layer((DenseLayer*)model->layers[0], num_features, 62);
    assert_dense_layer((DenseLayer*)model->layers[2], 62, 66);
    assert_dense_layer((DenseLayer*)model->layers[4], 66, 1);

    assert_ff_layer_cache(&model->tape[0], num_features, 62, batch_size, true, true);
    assert_ff_layer_cache(&model->tape[1], 62, 62, batch_size, false, false);
    assert_ff_layer_cache(&model->tape[2], 62, 66, batch_size, false, true);
    assert_ff_layer_cache(&model->tape[3], 66, 66, batch_size, false, false);
    assert_ff_layer_cache(&model->tape[4], 66, 1, batch_size, false, true);

    free_model(model);
}

int main(int argc, char** argv)
{
    test_create_and_free_model();
    return 0;
}
