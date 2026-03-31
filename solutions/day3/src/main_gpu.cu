#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <ctime>

#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)

__global__ void vectorAddGPU(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 100000000; // 100 million
    size_t size = N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize vectors
    for(int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaError_t err;

    err = cudaMalloc(&d_A, size);
    CUDA_CHECK(err);

    err = cudaMalloc(&d_B, size);
    CUDA_CHECK(err);

    err = cudaMalloc(&d_C, size);
    CUDA_CHECK(err);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    clock_t start = clock();

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; // trick that is equivalent to ceil(N / threadsPerBlock)
    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    err = cudaDeviceSynchronize();
    CUDA_CHECK(err);
    if (err != cudaSuccess) {
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
        exit(EXIT_FAILURE);
    }
    clock_t end = clock();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    double elapsedTime = ((double)(end - start))/CLOCKS_PER_SEC;

    printf("GPU Vector Add Time: %lf seconds\n", elapsedTime);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}