#include <assert.h>
#include "backprop.h"

void test_transpose_with_different_shape_dims()
{
    Matrix2D a[1], res[1];
    zeros_with_cache(a, 2, 3);
    zeros_with_cache(res, 3, 2);
    for (int i = 0; i < 6; i++)
        a->data[i] = i;

    transpose(a, res);

    double exp[] = { 0, 3, 1, 4, 2, 5 };
    for (int i = 0; i < 6; i++)
        assert(res->data[i] == exp[i]);

    free_matrix(a);
    free_matrix(res);
}

void test_transpose_of_transpose_is_identity()
{
    Matrix2D a[1], res[1], id[2];
    zeros_with_cache(a, 2, 3);
    zeros_with_cache(res, 3, 2);
    zeros_with_cache(id, 2, 3);
    random_normal(a, 0.0, 1.0);

    transpose(a, res);
    transpose(res, id);

    for (int i = 0; i < 6; i++)
        assert(id->data[i] == a->data[i]);

    free_matrix(a);
    free_matrix(res);
    free_matrix(id);
}

void test_matmul_with_identity_matrix_is_same_matrix()
{
    Matrix2D a[1], id[1], res[1];
    zeros_with_cache(id, 2, 2);
    id->data[0] = 1.0;
    id->data[3] = 1.0;

    zeros_with_cache(a, 2, 2);
    random_normal(a, 0.0, 1.0);

    zeros_with_cache(res, 2, 2);
    matmul(a, id, res, MATMUL_NN);

    for (int i = 0; i < 4; i++)
        assert(a->data[i] == res->data[i]);

    free_matrix(a);
    free_matrix(id);
    free_matrix(res);
}

void test_matmul_different_input_shapes_no_transpose()
{
    Matrix2D a1[1], a2[1], res[1];
    zeros_with_cache(a1, 2, 3);
    zeros_with_cache(a2, 3, 4);
    zeros_with_cache(res, 2, 4);
    for (int i = 0; i < 6; i++)
        a1->data[i] = i;
    for (int i = 0; i < 12; i++)
        a2->data[i] = i;

    matmul(a1, a2, res, MATMUL_NN);

    double exp[] = { 20, 23, 26, 29, 56, 68, 80, 92 };
    for (int i = 0; i < 8; i++)
        assert(res->data[i] == exp[i]);

    free_matrix(a1);
    free_matrix(a2);
    free_matrix(res);
}

void test_matmul_first_transposed()
{
    double data1[] = { 0, 3, 1, 4, 2, 5 };
    double data2[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    Matrix2D a1[1], a2[1], res[1];
    zeros_with_cache(a1, 3, 2);
    zeros_with_cache(a2, 3, 4);
    zeros_with_cache(res, 2, 4);
    for (int i = 0; i < 6; i++)
        a1->data[i] = data1[i];
    for (int i = 0; i < 12; i++)
        a2->data[i] = data2[i];

    matmul(a1, a2, res, MATMUL_TN);

    double exp[] = { 20, 23, 26, 29, 56, 68, 80, 92 };
    for (int i = 0; i < 8; i++)
        assert(res->data[i] == exp[i]);

    free_matrix(a1);
    free_matrix(a2);
    free_matrix(res);
}

void test_matmul_second_transposed()
{
    double data1[] = { 0, 1, 2, 3, 4, 5 };
    double data2[] = { 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11 };
    Matrix2D a1[1], a2[1], res[1];
    zeros_with_cache(a1, 2, 3);
    zeros_with_cache(a2, 4, 3);
    zeros_with_cache(res, 2, 4);
    for (int i = 0; i < 6; i++)
        a1->data[i] = data1[i];
    for (int i = 0; i < 12; i++)
        a2->data[i] = data2[i];

    matmul(a1, a2, res, MATMUL_NT);

    double exp[] = { 20, 23, 26, 29, 56, 68, 80, 92 };
    for (int i = 0; i < 8; i++)
        assert(res->data[i] == exp[i]);

    free_matrix(a1);
    free_matrix(a2);
    free_matrix(res);
}

void test_matmul_both_transposed()
{
    double data1[] = { 0, 3, 1, 4, 2, 5 };
    double data2[] = { 0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11 };
    Matrix2D a1[1], a2[1], res[1];
    zeros_with_cache(a1, 3, 2);
    zeros_with_cache(a2, 4, 3);
    zeros_with_cache(res, 2, 4);
    for (int i = 0; i < 6; i++)
        a1->data[i] = data1[i];
    for (int i = 0; i < 12; i++)
        a2->data[i] = data2[i];

    matmul(a1, a2, res, MATMUL_TT);

    double exp[] = { 20, 23, 26, 29, 56, 68, 80, 92 };
    for (int i = 0; i < 8; i++)
        assert(res->data[i] == exp[i]);

    free_matrix(a1);
    free_matrix(a2);
    free_matrix(res);
}

void test_elemmul()
{
    double data1[] = { 0, 1, 2, 3, 4, 5 };
    double data2[] = { 1, 2, 3, 4, 5, 6 };
    Matrix2D a1[1], a2[1], res[1];
    *a1 = (Matrix2D){ 3, 2, data1 };
    *a2 = (Matrix2D){ 3, 2, data2 };
    zeros_with_cache(res, 3, 2);

    elemmul(a1, a2, res);

    double exp[] = { 0, 2, 6, 12, 20, 30 };
    for (int i = 0; i < 6; i++)
        assert(res->data[i] == exp[i]);

    free_matrix(res);
}

void test_elemdiv()
{
    double data1[] = { 0, 1, 2, 3, 4, 5 };
    double data2[] = { 1, 2, 4, 6, 8, 10 };
    Matrix2D a1[1], a2[1], res[1];
    *a1 = (Matrix2D){ 3, 2, data1 };
    *a2 = (Matrix2D){ 3, 2, data2 };
    zeros_with_cache(res, 3, 2);

    elemdiv(a1, a2, res);

    double exp[] = { 0, 0.5, 0.5, 0.5, 0.5, 0.5 };
    for (int i = 0; i < 6; i++)
        assert(res->data[i] == exp[i]);

    free_matrix(res);
}

void test_elemsum()
{
    double data1[] = { 0, 1, 2, 3, 4, 5 };
    double data2[] = { 1, 2, 3, 4, 5, 6 };
    Matrix2D a1[1], a2[1], res[1];
    *a1 = (Matrix2D){ 3, 2, data1 };
    *a2 = (Matrix2D){ 3, 2, data2 };
    zeros_with_cache(res, 3, 2);

    elemsum(a1, a2, res);

    double exp[] = { 1, 3, 5, 7, 9, 11 };
    for (int i = 0; i < 6; i++)
        assert(res->data[i] == exp[i]);

    free_matrix(res);
}

void test_elemdiff()
{
    double data1[] = { 0, 1, 2, 3, 4, 5 };
    double data2[] = { 1, 2, 3, 4, 5, 6 };
    Matrix2D a1[1], a2[1], res[1];
    *a1 = (Matrix2D){ 3, 2, data1 };
    *a2 = (Matrix2D){ 3, 2, data2 };
    zeros_with_cache(res, 3, 2);

    elemdiff(a1, a2, res);

    double exp[] = { -1, -1, -1, -1, -1, -1 };
    for (int i = 0; i < 6; i++)
        assert(res->data[i] == exp[i]);

    free_matrix(res);
}

void test_batch_rowadd()
{
    double data1[] = { 0, 1, 2, 3, 4, 5 };
    double data2[] = { 1, 2 };
    Matrix2D a1[1], a2[1], res[1];
    *a1 = (Matrix2D){ 3, 2, data1 };
    *a2 = (Matrix2D){ 1, 2, data2 };
    zeros_with_cache(res, 3, 2);

    batch_rowadd(a1, a2, res);

    double exp[] = { 1, 3, 3, 5, 5, 7 };
    for (int i = 0; i < 6; i++)
        assert(res->data[i] == exp[i]);

    free_matrix(res);
}

void test_batch_colmean()
{
    double data1[] = { 0, 1, 2, 3, 4, 5 };
    Matrix2D a1[1], res[1];
    *a1 = (Matrix2D){ 3, 2, data1 };
    zeros_with_cache(res, 1, 2);

    batch_colmean(a1, res);

    double exp[] = { 2, 3 };
    for (int i = 0; i < 2; i++)
        assert(res->data[i] == exp[i]);

    free_matrix(res);
}

void test_batch_sum()
{
    double data1[] = { 0, 1, 2, 3, 4, 5 };
    Matrix2D a1[1], res[1];
    *a1 = (Matrix2D){ 3, 2, data1 };
    zeros_with_cache(res, 3, 2);

    batch_sum(a1, 2, res);

    double exp[] = { 2, 3, 4, 5, 6, 7 };
    for (int i = 0; i < 6; i++)
        assert(res->data[i] == exp[i]);

    free_matrix(res);
}

void test_batch_diff()
{
    double data1[] = { 0, 1, 2, 3, 4, 5 };
    Matrix2D a1[1], res[1];
    *a1 = (Matrix2D){ 3, 2, data1 };
    zeros_with_cache(res, 3, 2);

    batch_diff(a1, -2, res);

    double exp[] = { 2, 3, 4, 5, 6, 7 };
    for (int i = 0; i < 6; i++)
        assert(res->data[i] == exp[i]);

    free_matrix(res);
}

void test_batch_mul()
{
    double data1[] = { 0, 1, 2, 3, 4, 5 };
    Matrix2D a1[1], res[1];
    *a1 = (Matrix2D){ 3, 2, data1 };
    zeros_with_cache(res, 3, 2);

    batch_mul(a1, 2, res);

    double exp[] = { 0, 2, 4, 6, 8, 10 };
    for (int i = 0; i < 6; i++)
        assert(res->data[i] == exp[i]);

    free_matrix(res);
}

void test_batch_div()
{
    double data1[] = { 2, 4, 6, 8, 10, 12 };
    Matrix2D a1[1], res[1];
    *a1 = (Matrix2D){ 3, 2, data1 };
    zeros_with_cache(res, 3, 2);

    batch_div(a1, 2, res);

    double exp[] = { 1, 2, 3, 4, 5, 6 };
    for (int i = 0; i < 6; i++)
        assert(res->data[i] == exp[i]);

    free_matrix(res);
}

void test_batch_sqrt()
{
    double data1[] = { 1, 4, 9, 16, 25, 36 };
    Matrix2D a1[1], res[1];
    *a1 = (Matrix2D){ 3, 2, data1 };
    zeros_with_cache(res, 3, 2);

    batch_sqrt(a1, res);

    double exp[] = { 1, 2, 3, 4, 5, 6 };
    for (int i = 0; i < 6; i++)
        assert(res->data[i] == exp[i]);

    free_matrix(res);
}

void test_batch_max()
{
    double data1[] = { 0, 1, 2, 3, 4, 5 };
    Matrix2D a1[1], res[1];
    *a1 = (Matrix2D){ 3, 2, data1 };
    zeros_with_cache(res, 3, 2);

    batch_max(a1, 2, res);

    double exp[] = { 2, 2, 2, 3, 4, 5 };
    for (int i = 0; i < 6; i++)
        assert(res->data[i] == exp[i]);

    free_matrix(res);
}

void test_batch_geq()
{
    double data1[] = { 0, 1, 2, 3, 4, 5 };
    Matrix2D a1[1], res[1];
    *a1 = (Matrix2D){ 3, 2, data1 };
    zeros_with_cache(res, 3, 2);

    batch_geq(a1, 2, res);

    double exp[] = { 0, 0, 1, 1, 1, 1 };
    for (int i = 0; i < 6; i++)
        assert(res->data[i] == exp[i]);

    free_matrix(res);
}

void test_shuffle_rows_by_identity_perm_is_same_matrix()
{
    double data1[] = { 0, 1, 2, 3, 4, 5 };
    Matrix2D a1[1];
    *a1 = (Matrix2D){ 3, 2, data1 };

    int perm[] = { 0, 1, 2 };
    shuffle_rows(a1, perm);

    double exp[] = { 0, 1, 2, 3, 4, 5 };
    for (int i = 0; i < 6; i++)
        assert(a1->data[i] == exp[i]);
}

void test_shuffle_inverse_indices()
{
    double data1[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    Matrix2D a1[1];
    *a1 = (Matrix2D){ 10, 1, data1 };

    int perm[] = { 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
    shuffle_rows(a1, perm);

    double exp[] = { 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
    for (int i = 0; i < 10; i++)
        assert(a1->data[i] == exp[i]);
}

void test_shuffle_rows_according_to_perm()
{
    double data1[] = { 0, 1, 2, 3, 4, 5 };
    Matrix2D a1[1];
    *a1 = (Matrix2D){ 3, 2, data1 };

    int perm[] = { 2, 0, 1 };
    shuffle_rows(a1, perm);

    double exp[] = { 4, 5, 0, 1, 2, 3 };
    for (int i = 0; i < 6; i++)
        assert(a1->data[i] == exp[i]);
}

void test_generate_permutations()
{
    int n = 1000;
    int perm[n];
    for (int i = 0; i < n; i++)
        perm[i] = i;

    for (int test = 0; test < 100; test++)
    {
        int sum = 0;
        for (int i = 0; i < n; i++)
            sum += perm[i];
        assert(sum == n * (n - 1) / 2);
        permutate(perm, n);
    }
}

void test_shuffle_by_generated_permutations()
{
    int num_rows = 1000, num_cols = 10;
    Matrix2D data, data_old;
    zeros_with_cache(&data, num_rows, num_cols);
    zeros_with_cache(&data_old, num_rows, num_cols);
    random_normal(&data, 0.0, 1.0);

    int perm[num_rows];
    for (int i = 0; i < num_rows; i++)
        perm[i] = i;

    for (int test = 0; test < 100; test++)
    {
        permutate(perm, num_rows);
        copy(&data, &data_old);
        shuffle_rows(&data, perm);

        for (int row = 0; row < num_rows; row++)
            for (int col = 0; col < num_cols; col++)
                assert(data_old.data[perm[row] * num_cols + col] == data.data[row * num_cols + col]);
    }

    free_matrix(&data);
    free_matrix(&data_old);
}

void test_rand_normal_matrix_init()
{
    double exp_sigma = 1.0, exp_mu = 1.0;
    Matrix2D data;
    zeros_with_cache(&data, 1000, 10);
    random_normal(&data, exp_mu, exp_sigma);

    double mu = 0.0;
    for (int i = 0; i < data.num_rows * data.num_cols; i++)
        mu += data.data[i];
    mu /= data.num_rows * data.num_cols;

    double sigma = 0.0;
    for (int i = 0; i < data.num_rows * data.num_cols; i++)
        sigma += (mu - data.data[i]) * (mu - data.data[i]);
    sigma /= data.num_rows * data.num_cols;
    sigma = sqrt(sigma);

    assert(abs(mu - exp_mu) < 0.001);
    assert(abs(sigma - exp_sigma) < 0.001);
}

int main(int argc, char** argv)
{
    test_transpose_with_different_shape_dims();
    test_transpose_of_transpose_is_identity();
    test_matmul_with_identity_matrix_is_same_matrix();
    test_matmul_different_input_shapes_no_transpose();
    test_matmul_first_transposed();
    test_matmul_second_transposed();
    test_matmul_both_transposed();
    test_elemmul();
    test_elemdiv();
    test_elemsum();
    test_elemdiff();
    test_batch_rowadd();
    test_batch_colmean();
    test_batch_sum();
    test_batch_diff();
    test_batch_mul();
    test_batch_div();
    test_batch_sqrt();
    test_batch_max();
    test_batch_geq();
    test_shuffle_inverse_indices();
    test_shuffle_rows_by_identity_perm_is_same_matrix();
    test_shuffle_rows_according_to_perm();
    test_generate_permutations();
    test_shuffle_by_generated_permutations();
    test_rand_normal_matrix_init();
    return 0;
}
