#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_utils.h"
#include "matrix.h"

#define TILE_SIZE 16

__global__ void matrixAddKernel(const float *A, const float *B, float *C, const unsigned int height, const unsigned int width)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i = y * width + x;
    
    if (y >= height || x >= width) return;
    
    C[i] = A[i] + B[i];
}

int main() 
{
    size_t height = 1024;
    size_t width = 1024;
    size_t N = height * width;
	size_t bytes = N * sizeof(float);
    Matrix h_A, h_B, h_C;

    h_A.height = height;
    h_A.width = width;
	cudaMallocHost(&h_A.data, bytes);
    initMatrixValues(&h_A, -1.0f, 1.0f);

    h_B.height = height;
    h_B.width = width;
	cudaMallocHost(&h_B.data, bytes);
    initMatrixValues(&h_B, -1.0f, 1.0f);

	h_C.height = height;
    h_C.width = width;
	cudaMallocHost(&h_C.data, bytes);

	if (!verifyMatriciesForAdd(&h_A, &h_B))
    {
        cudaFreeHost(h_A.data);
        cudaFreeHost(h_B.data);
        cudaFreeHost(h_C.data);
        printf("Matrx dimensions do not match for addition, exiting..\n");
        exit(EXIT_FAILURE);
    } 

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

	CUDA_CHECK(cudaMemcpy(d_A, h_A.data, bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B.data, bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    // same as cuda::ceil_div(width / threadsPerBlock), cuda::ceil_div(height / threadsPerBlock)
    dim3 blocksPerGrid((int)(width + threadsPerBlock.x - 1) / threadsPerBlock.x, (int)(height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, (unsigned int) h_C.height, (unsigned int) h_C.width);

    Matrix *h_cpu_C = matrixAdd(&h_A, &h_B);

	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(h_C.data, d_C, bytes, cudaMemcpyDeviceToHost));

    if (compareMatricies(h_cpu_C, &h_C)) 
    {
        printf("GPU and CPU Matricies match! :D\n");
    }
    else
    {
        printf("GPU and CPU Matricies do NOT match! D:\n");
    }
        
	
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaFreeHost(h_A.data);
    cudaFreeHost(h_B.data);
    cudaFreeHost(h_C.data);

    free(h_cpu_C->data);
    free(h_cpu_C);
    
    return EXIT_SUCCESS;
}