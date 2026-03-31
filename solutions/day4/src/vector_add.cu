#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "cuda_utils.h"

__global__ void vectorAdd(float *A, float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

void cpuVectorAdd(float *A, float *B, float *C, int N)
{
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
}

bool is_equal(float *d_arr, float *h_arr, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (fabsf(d_arr[i] - h_arr[i]) > 1E-8)
        {
            return false;
        }
    }

    return true;
}

int main() 
{
    int N = 100000000; // 100 million (10^8)

    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C, *h_C_result_comparison;

    h_A = (float*) malloc(N * sizeof(float));
    h_B = (float*) malloc(N * sizeof(float));
    h_C = (float*) malloc(N * sizeof(float));
    h_C_result_comparison = (float*) malloc(N * sizeof(float));

    for(int i = 0; i < N; i++) 
    {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    cudaError_t err;

    CUDA_CHECK(cudaMalloc(&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    int blockDim = 256;
    int gridDim = (N + blockDim - 1) / blockDim; // same as cuda::ceil_div(N / blockDim)

    vectorAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cpuVectorAdd(h_A, h_B, h_C_result_comparison, N);

    err = cudaDeviceSynchronize();
    CUDA_CHECK(err);
    if (err != cudaSuccess)
    {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
        free(h_C_result_comparison);
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    if (!is_equal(h_C, h_C_result_comparison, N)){
        printf("Result Arrays are NOT equal! :(\n");
    }
    else
    {
        printf("Result Arrays are equal! :)\n");
    }

    printf("Sample Operations\n");

    size_t arr_size = 3;
    int arr[4] = {0, 1, N-1};
    for (int i = 0; i < arr_size; i++)
    {
        printf("i=%d: %f + %f = %f\n", arr[i], h_A[arr[i]], h_B[arr[i]], h_C[arr[i]]);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_result_comparison);
    
    return 0;
}