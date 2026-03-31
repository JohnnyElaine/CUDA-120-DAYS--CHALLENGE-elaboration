#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_utils.h"

#define THREAD_BLOCK_SIZE 16

// row major matrix
typedef struct {
    size_t height;
    size_t width;
    float *data;
} Matrix;

__global__ void matrixAddGPU(const float *A, const float *B, float *C, const size_t height, const size_t width)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = row * width + col;
    if (row < height && col < width)
    {
        C[i] = A[i] + B[i];
    }
}

Matrix *matrixAddCPU(const Matrix *A, const Matrix *B)
{
    Matrix *C = (Matrix*) malloc(sizeof(Matrix));
    C->height = A->height;
    C->width = A->width;
    size_t N = C->height * C->width;
	C->data = (float*) malloc(N * sizeof(float));

    for (size_t row = 0; row < A->height; row++)
    {
        for (size_t col = 0; col < A->width; col++)
        {
            int i = row * C->width + col;
            C->data[i] = A->data[i] + B->data[i];
        }
    }

    return C;
}

void initMatrixValues(Matrix *M, float base)
{
    for (size_t row = 0; row < M->height; row++)
    {
        for (size_t col = 0; col < M->width; col++)
        {
            size_t i = row * M->width + col;
            M->data[i] = i * base;
        }
    }
}

void compareMatricies(Matrix *A, Matrix *B)
{
    for (size_t row = 0; row < A->height; row++)
    {
        for (size_t col = 0; col < A->width; col++)
        {
            size_t i = row * A->width + col;
            if (fabsf(A->data[i] - B->data[i]) > 1E-8)
            {
                printf("Matricies at row=%d, column=%d do not match!\n", row, col);
            }
        }
    }
}

bool verifyMatricies(Matrix *A, Matrix *B){
	if (A->height != B->height || A->width != B->width) 
    {
        printf("Matrix sizes does not match!\n");
		return false;
    }

	return true;
}

void printMatrix(Matrix *M)
{
	for (size_t row = 0; row < M->height; row++)
    {
        for (size_t col = 0; col < M->width; col++)
        {
            size_t i = row * M->width + col;
			printf("%f\t", M->data[i]);
        }
		printf("\n");
    }
}

int main() 
{
    size_t height = 8;
    size_t width = 8;
    size_t N = height * width;
	size_t bytes = N * sizeof(float);
    Matrix h_A, h_B, h_C;

    h_A.height = height;
    h_A.width = width;
	cudaMallocHost(&h_A.data, bytes);
    initMatrixValues(&h_A, 2.0f);

    h_B.height = height;
    h_B.width = width;
	cudaMallocHost(&h_B.data, bytes);
    initMatrixValues(&h_B, 4.0f);

	h_C.height = height;
    h_C.width = width;
	cudaMallocHost(&h_C.data, bytes);

	if (!verifyMatricies(&h_A, &h_B)) exit(EXIT_FAILURE);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

	CUDA_CHECK(cudaMemcpy(d_A, h_A.data, bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B.data, bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
    // same as cuda::ceil_div(width / threadsPerBlock)
    // same as cuda::ceil_div(height / threadsPerBlock)
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, h_C.height, h_C.width);

    Matrix *h_cpu_C = matrixAddCPU(&h_A, &h_B);

	cudaDeviceSynchronize();

	CUDA_CHECK(cudaMemcpy(h_C.data, d_C, bytes, cudaMemcpyDeviceToHost));

    compareMatricies(h_cpu_C, &h_C);

	printf("Matrix A:\n");
	printMatrix(&h_A);
	printf("Matrix B:\n");
	printMatrix(&h_B);
	printf("Result Matrix C (A + B):\n");
	printMatrix(&h_C);

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