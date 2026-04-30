#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_utils.h"
#include "matrix.h"
#include "matrix_operations_kernels.cuh"

int main() 
{
    // TODO measure performance and results of both kernel implementations

    // Define matricies
    Matrix h_A, h_B, h_C;

    h_A.height = 1024;
    h_A.width = 16;
    const size_t h_A_bytes = h_A.height * h_A.width * sizeof(float);
	cudaMallocHost(&h_A.data, h_A_bytes);
    initMatrixValues(&h_A, -1000.0f, 1000.0f);

    h_B.height = 16;
    h_B.width = 512;
    const size_t h_B_bytes = h_B.height * h_B.width * sizeof(float);
	cudaMallocHost(&h_B.data, h_B_bytes);
    initMatrixValues(&h_B, -1000.0f, 1000.0f);

   
	h_C.height = h_A.height;
    h_C.width = h_B.width;
    const size_t h_C_bytes = h_C.height * h_C.width * sizeof(float);
	cudaMallocHost(&h_C.data, h_C_bytes);
    

	if (!verifyMatriciesForMult(&h_A, &h_B)) {
        cudaFreeHost(h_A.data);
        cudaFreeHost(h_B.data);
        cudaFreeHost(h_C.data);
        printf("Matrx dimensions do not match for multiplication, exiting..\n");
        exit(EXIT_FAILURE);
    } 

    // Copy matricies from host to device
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, h_A_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, h_B_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, h_C_bytes));

	CUDA_CHECK(cudaMemcpy(d_A, h_A.data, h_A_bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B.data, h_B_bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    // same as cuda::ceil_div(width / threadsPerBlock), cuda::ceil_div(height / threadsPerBlock)
    dim3 blocksPerGrid((int)(h_C.width + threadsPerBlock.x - 1) / threadsPerBlock.x, (int)(h_C.height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMultImprovedKernel32FP<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, (unsigned int) h_A.height, (unsigned int) h_A.width, (unsigned int) h_B.width);

    Matrix *h_cpu_C = matrixMult(&h_A, &h_B);

	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(h_C.data, d_C, h_C_bytes, cudaMemcpyDeviceToHost));

    if (compareMatricies(h_cpu_C, &h_C))
    {
        printf("Matricies match!.\n");
    } 
	
    // Cleanup allocated memory
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