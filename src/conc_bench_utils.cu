/**
 * @file conc_bench_utils.cu
 * @date 8/03/2021
 * @author Mirco De Marchi
 * @brief Source of Concurrent Benchmark utils.
 */

/*
 * Apologies to whoever will have to read this code, I just discovered
 * precompiler macros and I went crazy with it..
 */
#include <iostream>
#include "Timer.cuh"
#include "CheckError.cuh"

#include "conc_bench_utils.cuh"

using namespace timer;

__global__ static void sum_gpu_left(float *matrix, const int N);
__global__ static void sum_gpu_right(float *matrix, const int N);

void matrix_supplier(std::vector<float> (&v))
{
    // -------------------------------------------------------------------------
    // MATRIX INITILIZATION
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(1, 100);

    for (int i = 0; i < N; i++)
    {
        float temp = distribution(generator);
        v.push_back(temp);
    }
}

void sum_cpu(std::vector<float> (&v), int sum_id) 
{
    float *arr = &v[0];
    if (sum_id % 2 != 0)
    {
        sum_cpu_right(arr, N);
    }
    else 
    {
        sum_cpu_left(arr, N);
    }
}

void sum_gpu(std::vector<float> (&v), float *d_matrix_device, int sum_id) 
{
    const int grid = N / BLOCK_SIZE_X;
    float *d_matrix = &v[0];
    // -------------------------------------------------------------------------
    // DEVICE INIT
    dim3 DimGrid(grid, 1, 1);
    if (N % grid)
        DimGrid.x++;
    dim3 DimBlock(BLOCK_SIZE_X, 1, 1);

    // -------------------------------------------------------------------------
    // EXECUTION
    if (sum_id % 2 != 0)
    {
#if COPY
        SAFE_CALL(cudaMemcpy(d_matrix_device, d_matrix, N * sizeof(int),
                                cudaMemcpyHostToDevice));
        sum_gpu_left<<<DimGrid, DimBlock>>>(d_matrix_device, N);
        CHECK_CUDA_ERROR
        SAFE_CALL(cudaMemcpy(d_matrix, d_matrix_device, N * sizeof(int),
                                cudaMemcpyDeviceToHost));
#else
        sum_gpu_left<<<DimGrid, DimBlock>>>(d_matrix, N);
#endif
#if UNIFIED
        // This macro includes cudaDeviceSynchronize(), which makes the program
        // work on the data in lockstep
        CHECK_CUDA_ERROR
#endif
    }
    else
    {
#if COPY
        SAFE_CALL(cudaMemcpy(d_matrix_device, d_matrix, N * sizeof(int),
                                cudaMemcpyHostToDevice));
        sum_gpu_right<<<DimGrid, DimBlock>>>(d_matrix_device, N);
        CHECK_CUDA_ERROR
        SAFE_CALL(cudaMemcpy(d_matrix, d_matrix_device, N * sizeof(int),
                                cudaMemcpyDeviceToHost));
#else
        sum_gpu_right<<<DimGrid, DimBlock>>>(d_matrix, N);
#endif
#if UNIFIED
        CHECK_CUDA_ERROR
#endif
    }

#if ZEROCOPY
    // Synchronization needed to avoid race conditions (after the CPU and 
    // GPU have done their sides, we need to sync)
    CHECK_CUDA_ERROR
#endif
}

void init_gpu(float **d_matrix_device) 
{
#if ZEROCOPY
    cudaSetDeviceFlags(cudaDeviceMapHost);
#endif
    SAFE_CALL(cudaMalloc(d_matrix_device, N * sizeof(float)));
}

void sum_cpu_right(float *d_matrix, const int N)
{
#if OPENMP
#pragma omp parallel for
#endif
    for (int j = N / 2; j < N; j++)
    {
        if (j % 2 == 0)
        {
            //__sync_fetch_and_add(&d_matrix[j], 1);
            for (int r = 0; r < 2; r++)
            {
                d_matrix[j] = sqrt(d_matrix[j] * (d_matrix[j] / 2.3));
            }
            // printf("cpu right: %d\n", j);
        }
    }
}

void sum_cpu_left(float *d_matrix, const int N)
{
#if OPENMP
#pragma omp parallel for
#endif
    for (int j = 0; j < N / 2; j++)
    {
        if (j % 2 != 0)
        {
            //__sync_fetch_and_add(&d_matrix[j], 1);
            for (int r = 0; r < 2; r++)
            {
                d_matrix[j] = sqrt(d_matrix[j] * (d_matrix[j] / 2.3));
            }
            // printf("cpu left: %d\n", j);
        }
    }
}

void sum_cpu_only(float *matrix)
{
#if CPU
    for (int i = 0; i < SUMS; i++)
    {
        if (i % 2 != 0)
        {
            for (int j = 0; j < N / 2; j++)
            {
                if (j % 2 != 0)
                {
                    float temp = 2.0 * sqrt(matrix[j] + matrix[j + N / 2]);
                    for (int f = 0; f < 2; f++)
                    {
                        temp /= float(f) + sqrt(3.14159265359 * temp) / 0.7;
                        temp *= 1.6;
                    }
                    matrix[j] = temp;
                }
            }
            for (int j = N / 2; j < N; j++)
            {
                if (j % 2 == 0)
                {
                    for (int r = 0; r < 2; r++)
                    {
                        matrix[j] = sqrt(matrix[j] * (matrix[j] / 2.3));
                    }
                }
            }
        }
        else
        {
            for (int j = N / 2; j < N; j++)
            {
                if (j % 2 == 0)
                {
                    float temp = 2.0 * sqrt(matrix[j] + matrix[j - N / 2]);
                    for (int f = 0; f < 2; f++)
                    {
                        temp /= float(f) + sqrt(3.14159265359 * temp) / 0.7;
                        temp *= 1.6;
                    }
                    matrix[j] = temp;
                }
            }
            for (int j = 0; j < N / 2; j++)
            {
                if (j % 2 != 0)
                {
                    for (int r = 0; r < 2; r++)
                    {
                        matrix[j] = sqrt(matrix[j] * (matrix[j] / 2.3));
                    }
                }
            }
        }
#if PRINT
        printf("RUN %d\n", i);
        printf("Values from index %d to %d\n", FROM_debug, TO_debug);
        printf("H: ");
        for (int i = FROM_debug; i < TO_debug; i++)
        {
            if (i % (N / 2) == 0)
                printf("| ");
            printf("%.2f ", matrix[i]);
        }
        printf("\n");
#endif
    }
#else
    for (int i = 0; i < SUMS; i++)
    {
        for (int j = 0; j < N / 2; j++)
        {
            if (j % 2 != 0)
            {
                float temp = 2.0 * sqrt(matrix[j] + matrix[j + N / 2]);
                for (int f = 0; f < 2; f++)
                {
                    temp /= float(f) + sqrt(3.14159265359 * temp) / 0.7;
                    temp *= 1.6;
                }
                matrix[j] = temp;
            }
        }
        for (int j = N / 2; j < N; j++)
        {
            if (j % 2 == 0)
            {
                float temp = 2.0 * sqrt(matrix[j] + matrix[j + N / 2]);
                for (int f = 0; f < 2; f++)
                {
                    temp /= float(f) + sqrt(3.14159265359 * temp) / 0.7;
                    temp *= 1.6;
                }
                matrix[j] = temp;
            }
        }
    }
#endif
}

__global__ static void sum_gpu_left(float *matrix, const int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N / 2)
    {
        if (row % 2 != 0)
        {
            float temp = 2.0 * sqrt(matrix[row] + matrix[row + N / 2]);
            for (int f = 0; f < 2; f++)
            {
                temp /= float(f) + sqrt(3.14159265359 * temp) / 0.7;
                temp *= 1.6;
            }
            matrix[row] = temp;
        }
    }
}

__global__ static void sum_gpu_right(float *matrix, const int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N / 2 && row < N)
    {
        if (row % 2 == 0)
        {
            float temp = 2.0 * sqrt(matrix[row] + matrix[row - N / 2]);
            for (int f = 0; f < 2; f++)
            {
                temp /= float(f) + sqrt(3.14159265359 * temp) / 0.7;
                temp *= 1.6;
            }
            matrix[row] = temp;
        }
    }
}