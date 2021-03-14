/**
 * @file conc_bench_utils.cuh
 * @date 8/03/2021
 * @author Mirco De Marchi
 * @brief Header of Concurrent Benchmark utils.
 */

#ifndef CONC_BENCH_UTILS_H_
#define CONC_BENCH_UTILS_H_

#include <vector>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

#include <stdio.h>

#include <omp.h>

// Set PRINT to 1 for debug output
#define PRINT 0
#define FROM_debug 0
#define TO_debug 16

// Set ZEROCOPY to 1 to use Zero Copy Memory Mode, UNIFIED to 1 to use Unified
// Memory, COPY to 1 to use Copy (only one of them can be 1, others must be 0)
#define ZEROCOPY 0
#define UNIFIED 0
#define COPY 1

// Set RESULTCHECK to 1 to verify the result with a single CPU thread
#define RESULTCHECK 1

// Set CPU to 1 to use the CPU concurrently
#define CPU 1
// Set OPENMP to 1 to use more than 1 thread for the CPU
#define OPENMP 1

const int POW = 23; // Maximum is 30, anything higher and the system will use
                    // swap, making the Cuda kernels crash
const int RUNS = 1; // How many times the benchmark is run
const int SUMS = 10; // As CPU and GPU work on either the left side or right 
                     // side, this number indicates how many "side swaps" 
                     // there will be
const int BLOCK_SIZE_X = 1024; // Cuda Block Size

const unsigned int N = 2 << POW;

void matrix_supplier(std::vector<float> (&v));
void sum_cpu(std::vector<float> (&v), int sum_id);
void sum_gpu(std::vector<float> (&v), float *d_matrix_device, int sum_id);
void init_gpu(float **d_matrix_device);

void sum_cpu_right(float *d_matrix, const int N);
void sum_cpu_left(float *d_matrix, const int N);
void sum_cpu_only(float *matrix);

#endif