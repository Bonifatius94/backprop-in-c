#include <math.h>
#include <stddef.h>
#include <assert.h>
#include <immintrin.h>

#pragma once

// #define MATMUL_AVX512 1
#define MATMUL_AVX256 1

/* Generate a uniform random double between [0, 1]. */
double random_uniform_double()
{
    return (double)rand() / RAND_MAX;
}

/* Generate a uniform random integer between [0, n-1]. */
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

/* Shuffle a pre-initialized permutation with Fisher-Yates. */
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

/* 2D matrix with a given amount of rows and columns. */
typedef struct Matrix2D {
    int num_rows;
    int num_cols;
    double* data;
    double* cache;
} Matrix2D;

/* Assign an empty matrix. */
#define EMPTY_MATRIX ((Matrix2D){0, 0, NULL, NULL})

/* Flags for matrix multiplication, indicating whether
   input matrices need to be viewed as transposed. */
typedef enum MatmulFlags {
    MATMUL_NN, MATMUL_TN, MATMUL_NT, MATMUL_TT
} MatmulFlags;

#if MATMUL_AVX512
    #define ALIGN_BYTES 64
#elif MATMUL_AVX256
    #define ALIGN_BYTES 32
#else
    #define ALIGN_BYTES 1
#endif
#define align_to_register(n) ((int)(ceil((n) / (double)ALIGN_BYTES)) * ALIGN_BYTES)

/* Initialize an unallocated matrix with zeros according to the shape. */
void zeros(Matrix2D a1[1], int num_rows, int num_cols)
{
    a1->num_rows = num_rows;
    a1->num_cols = num_cols;
    int size = align_to_register(num_rows * num_cols);
    a1->data = (double*)calloc(size, sizeof(double));
    a1->cache = NULL;
}

/* Initialize an unallocated matrix with zeros according to the shape.
   Moreover, initialize an equally sized cache for faster matrix ops. */
void zeros_with_cache(Matrix2D a1[1], int num_rows, int num_cols)
{
    a1->num_rows = num_rows;
    a1->num_cols = num_cols;
    int size = align_to_register(num_rows * num_cols);
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

    int size = align_to_register(a1->num_rows * a1->num_cols);
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

/* Compute the dot product of two line vectors. */
double dot_product(Matrix2D v1[1], Matrix2D v2[1])
{
    assert(v1->num_rows == 1 && v2->num_rows == 1);
    assert(v1->num_cols == v2->num_cols);

#if MATMUL_AVX512

    /* compute packed madd ops with 256-bit SIMD */
    int i = 0;
    __m512d sum = _mm512_setzero_pd();
    for (; i < v1->num_cols / 8 * 8; i += 8)
    {
        __m512d a_vec = _mm512_loadu_pd(&v1->data[i]);
        __m512d b_vec = _mm512_loadu_pd(&v2->data[i]);
        sum = _mm512_fmadd_pd(a_vec, b_vec, sum);
        /* madd := a * b + c */
    }

    /* unroll remainder not aligned with 8 doubles */
    double result = 0;
    for (; i < v1->num_cols; i++)
        result += v1->data[i] * v2->data[i];

    /* collect results as sum */
    for (i = 0; i < 8; ++i)
        result += ((double*)&sum)[i];
    return result;

#elif MATMUL_AVX256

    /* compute packed madd ops with 256-bit SIMD */
    int i = 0;
    __m256d sum = _mm256_setzero_pd();
    for (; i < v1->num_cols / 4 * 4; i += 4)
    {
        __m256d a_vec = _mm256_loadu_pd(&v1->data[i]);
        __m256d b_vec = _mm256_loadu_pd(&v2->data[i]);
        sum = _mm256_fmadd_pd(a_vec, b_vec, sum);
        /* madd := a * b + c */
    }

    /* unroll remainder not aligned with 4 doubles */
    double result = 0;
    for (; i < v1->num_cols; i++)
        result += v1->data[i] * v2->data[i];

    /* collect results as sum */
    for (i = 0; i < 4; ++i)
        result += ((double*)&sum)[i];
    return result;

#else /* unoptimized version */

    double sum = 0.0;
    for (int i = 0; i < v1->num_cols; i++)
        sum += v1->data[i] * v2->data[i];
    return sum;

#endif
}

/* Transpose the given matrix a1 and write the result to res. */
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

/* Elementwise add two matrices a1, a2 and write the result to res. */
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

/* Elementwise subtract matrix a2 from a1 and write the result to res. */
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

/* Elementwise multiply two matrices a1, a2 and write the result to res. */
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

/* Elementwise divide matrix a1 by a2 and write the result to res. */
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

/* Add the row vector to each row of the matrix a1 and write the result to res. */
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

/* Aggregate the mean of each column of a1 and write the result to res. */
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

/* Add a single value to all elements in matrix a1 and write the result to res. */
void batch_sum(const Matrix2D a1[1], double summand, Matrix2D res[1])
{
    assert(a1->num_rows == res->num_rows);
    assert(a1->num_cols == res->num_cols);

    /* TODO: use SIMD to speed this up */
    for (int i = 0; i < a1->num_rows * a1->num_cols; i++)
        res->data[i] = a1->data[i] + summand;
}

/* Subtract a single value from all elements in matrix a1 and write the result to res. */
void batch_diff(const Matrix2D a1[1], double diff, Matrix2D res[1])
{
    batch_sum(a1, -diff, res);
}

/* Multiply all elements in matrix a1 by a single value and write the result to res. */
void batch_mul(const Matrix2D a1[1], double factor, Matrix2D res[1])
{
    assert(a1->num_rows == res->num_rows);
    assert(a1->num_cols == res->num_cols);

    /* TODO: use SIMD to speed this up */
    for (int i = 0; i < a1->num_rows * a1->num_cols; i++)
        res->data[i] = a1->data[i] * factor;
}

/* Divide all elements in matrix a1 by a single value and write the result to res. */
void batch_div(const Matrix2D a1[1], double factor, Matrix2D res[1])
{
    assert(factor != 0.0);
    factor = 1 / factor;
    batch_mul(a1, factor, res);
}

/* Compute the sqrt() of all elements in matrix a1 and write the result to res. */
void batch_sqrt(const Matrix2D a1[1], Matrix2D res[1])
{
    assert(a1->num_rows == res->num_rows);
    assert(a1->num_cols == res->num_cols);

    /* TODO: use SIMD to speed this up */
    for (int i = 0; i < a1->num_rows * a1->num_cols; i++)
        res->data[i] = sqrt(a1->data[i]);
}

/* Cap all elements in matrix a1 by a minimum value and write the result to res. */
void batch_max(const Matrix2D a1[1], double min, Matrix2D res[1])
{
    assert(a1->num_rows == res->num_rows);
    assert(a1->num_cols == res->num_cols);

    /* TODO: use SIMD to speed this up */
    for (int i = 0; i < a1->num_rows * a1->num_cols; i++)
        res->data[i] = a1->data[i] >= min ? a1->data[i] : min;
}

/* Indicate for all elements in matrix a1 whether the value is greater of equal
   than a given minimum value and write the result to res. */
void batch_geq(const Matrix2D a1[1], double min, Matrix2D res[1])
{
    assert(a1->num_rows == res->num_rows);
    assert(a1->num_cols == res->num_cols);

    for (int i = 0; i < a1->num_rows * a1->num_cols; i++)
        res->data[i] = a1->data[i] >= min ? 1.0 : 0.0;
}

/* Shuffle the rows according to the given permutation (in-place). */
void shuffle_rows(Matrix2D a1[1], const int perm[])
{
    double* temp_data = (double*)malloc(a1->num_rows * a1->num_cols * sizeof(double));

    for (int i = 0; i < a1->num_rows; i++)
        for (int j = 0; j < a1->num_cols; j++)
            temp_data[i * a1->num_cols + j] = a1->data[perm[i] * a1->num_cols + j];

    for (int i = 0; i < a1->num_rows; i++)
        for (int j = 0; j < a1->num_cols; j++)
            a1->data[i * a1->num_cols + j] = temp_data[i * a1->num_cols + j];

    free(temp_data);
}

/* Free the data managed by the matrix. */
void free_matrix(Matrix2D a1[1])
{
    if (a1->data != NULL)
        free(a1->data);
    if (a1->cache != NULL)
        free(a1->cache);
}
