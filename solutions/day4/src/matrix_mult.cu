#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "cuda_utils.h"
#include "matrix.h"

#define TILE_SIZE 16


/**
 * @brief Performs matrix multiplication on the GPU: C = A * B
 * 
 * Computes the product of a MxN matrix (A) and a NxK matrix (B), 
 * storing the result in a MxK matrix (C).
 *
 * @note All matrices (A, B, and C) must be stored in **row-major** order 
 *       as flattened 1D arrays in memory.
 *
 * @param[in]  A  Pointer to the first input matrix (MxN). Must be allocated on the device.
 * @param[in]  B  Pointer to the second input matrix (NxK). Must be allocated on the device.
 * @param[out] C  Pointer to the output matrix (MxK). Must be allocated on the device.
 * @param[in]  M  Number of rows in matrix A and matrix C.
 * @param[in]  N  Number of columns in matrix A and rows in matrix B.
 * @param[in]  K  Number of columns in matrix B and matrix C.
 *
 * @par Grid Configuration:
 * The kernel must be launched with a 2D grid and 2D thread blocks:
 * @endcode
 */
__global__ void matrixMultKernel(const float* A, const float* B, float* C, const unsigned int M, const unsigned int N, const unsigned int K)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= K || y >= M) return;

    float sum = 0.0f;
    unsigned int a_idx;
    unsigned int b_idx;

    // row * column
    for (int elem = 0; elem < N; ++elem)
    {
        a_idx = y * N + elem; // A[y, elem_idx]
        b_idx = elem * K + x; // B[elem_idx, x]
        sum += A[a_idx] * B[b_idx];
    }

    unsigned int c_idx = y * K + x;
    C[c_idx] = sum;
}

__device__ __forceinline__ void loadTileA(const float *A, float sm[TILE_SIZE][TILE_SIZE], const int y, const unsigned int M, const unsigned int N, const unsigned int tile_idx)
{
    const unsigned int col_A = tile_idx * TILE_SIZE + threadIdx.x;
    if (y < M && col_A < N)
        sm[threadIdx.y][threadIdx.x] = A[y * N + col_A];
    else
        sm[threadIdx.y][threadIdx.x] = 0.0f;
}

__device__ __forceinline__ void loadTileB(const float *B, float sm[TILE_SIZE][TILE_SIZE], const int x, const int N, const int K, const int tile_idx)
{
    const unsigned int row_B = tile_idx * TILE_SIZE + threadIdx.y;
    if (row_B < N && x < K)
        sm[threadIdx.y][threadIdx.x] = B[row_B * K + x];
    else
        sm[threadIdx.y][threadIdx.x] = 0.0f;
}

__device__ __forceinline__ void dotProductTile(float *sum, const int A_row, const int B_col, const float sm_A[TILE_SIZE][TILE_SIZE], const float sm_B[TILE_SIZE][TILE_SIZE])
{
    for (int i = 0; i < TILE_SIZE; ++i)
    {
        *sum += sm_A[A_row][i] * sm_B[i][B_col];
    }
}

__global__ void matrixMultImprovedKernel(const float* A, const float* B, float* C, const unsigned int M, const unsigned int N, const unsigned int K) 
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    __shared__ float sm_A[TILE_SIZE][TILE_SIZE];
    __shared__ float sm_B[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (unsigned int tile_idx = 0; tile_idx <  (N + TILE_SIZE - 1) / TILE_SIZE; ++tile_idx)
    {
        // Loading Phase: Load 2D Tiles into shared memory
        loadTileA(A, sm_A, y, M, N, tile_idx);
        loadTileB(B, sm_B, x, N, K, tile_idx);

        __syncthreads();
        
        // Computation Phase: Perform the dot product (A-row x B-col) for the current part of the tile
        // A[row,:] * B[:, col]
        
        dotProductTile(&sum ,threadIdx.y, threadIdx.x, sm_A, sm_B);
        
        __syncthreads();
    }

    if (x >= K || y >= M) return;
    C[y * K + x] = sum;
}

int main() 
{
    // TODO measure performance and results of both kernel implementations
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

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, h_A_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, h_B_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, h_C_bytes));

	CUDA_CHECK(cudaMemcpy(d_A, h_A.data, h_A_bytes, cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_B, h_B.data, h_B_bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    // same as cuda::ceil_div(width / threadsPerBlock), cuda::ceil_div(height / threadsPerBlock)
    dim3 blocksPerGrid((int)(h_C.width + threadsPerBlock.x - 1) / threadsPerBlock.x, (int)(h_C.height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMultKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, (unsigned int) h_A.height, (unsigned int) h_A.width, (unsigned int) h_B.width);

    Matrix *h_cpu_C = matrixMult(&h_A, &h_B);

	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(h_C.data, d_C, h_C_bytes, cudaMemcpyDeviceToHost));

    if (compareMatricies(h_cpu_C, &h_C))
    {
        printf("Matricies match!.\n");
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