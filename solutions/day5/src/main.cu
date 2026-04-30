#include <random>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_utils.h"

#define EPSILON 1e-3

void initHostArr(float *A, float *B, int N) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    for (int i = 0; i < N; ++i) {
        A[i] =  dist(gen);
        B[i] =  dist(gen);
    }
}

__global__ void compareResults(float *A, float *B, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    if (fabsf(A[idx] - B[idx]) > EPSILON) {
        printf("idx: %d not equals: A[%d]=%f, B[%d]=%f", idx, idx, A[idx], idx, B[idx]);
    }
}


__global__ void divergenceKernel(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;

    if (A[idx] > 0) {
        C[idx] = A[idx] + B[idx];
    } else {
        C[idx] = A[idx] - B[idx];
    }
}

__global__ void noDivergenceKernel(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) return;
    float sign = A[idx] > 0 ? 1.0f : -1.0f;
    C[idx] = A[idx] + B[idx] * sign;
}


int main() {
    int N = 1e6;
    float *A, *B, *d_A, *d_B, *d_C1, *d_C2;

    cudaMallocHost(&A, N * sizeof(float));
    cudaMallocHost(&B, N * sizeof(float));

    initHostArr(A, B, N);
    
    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C1, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C2, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // --- CUDA Event creation for timers ---
    cudaEvent_t start, stop;
    float time_divergence, time_no_divergence;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ------------------------
    // 1. Measure Divergence Kernel
    // ------------------------
    cudaEventRecord(start, 0);                     // Record start time
    divergenceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C1, N);
    cudaEventRecord(stop, 0);                     // Record stop time
    cudaEventSynchronize(stop);                   // Wait for kernel to finish
    cudaEventElapsedTime(&time_divergence, start, stop);
    printf("divergenceKernel time: %f ms\n", time_divergence);

    // ------------------------
    // 2. Measure No-Divergence Kernel
    // ------------------------
    cudaEventRecord(start, 0);
    noDivergenceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C2, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_no_divergence, start, stop);
    printf("noDivergenceKernel time: %f ms\n", time_no_divergence);

    // --- Speedup Calculation ---
    if (time_no_divergence > 0) {
        float speedup = time_divergence / time_no_divergence;
        printf("Speedup (divergence/no_divergence): %.2fx\n", speedup);
        printf("The noDivergenceKernel is %.2fx faster.\n", speedup);
    }

    cudaDeviceSynchronize();
    compareResults<<<blocksPerGrid, threadsPerBlock>>>(d_C1, d_C2, N);
    
    cudaFreeHost(A);
    cudaFreeHost(B);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C1);
    cudaFree(d_C2);

    return EXIT_SUCCESS;
}