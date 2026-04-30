#include "matrix_operations_kernels.cuh"

__global__ void matrixAddKernel(const float *A, const float *B, float *C, const unsigned int height, const unsigned int width)
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i = y * width + x;
    
    if (y >= height || x >= width) return;
    
    C[i] = A[i] + B[i];
}

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
__global__ void matrixMultKernel32FP(const float* A, const float* B, float* C, const unsigned int M, const unsigned int N, const unsigned int K)
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

__device__ __forceinline__ void loadTileA32FP(const float *A, float sm[TILE_SIZE][TILE_SIZE], const int y, const unsigned int M, const unsigned int N, const unsigned int tile_idx)
{
    const unsigned int col_A = tile_idx * TILE_SIZE + threadIdx.x;
    if (y < M && col_A < N)
        sm[threadIdx.y][threadIdx.x] = A[y * N + col_A];
    else
        sm[threadIdx.y][threadIdx.x] = 0.0f;
}

__device__ __forceinline__ void loadTileB32FP(const float *B, float sm[TILE_SIZE][TILE_SIZE], const int x, const int N, const int K, const int tile_idx)
{
    const unsigned int row_B = tile_idx * TILE_SIZE + threadIdx.y;
    if (row_B < N && x < K)
        sm[threadIdx.y][threadIdx.x] = B[row_B * K + x];
    else
        sm[threadIdx.y][threadIdx.x] = 0.0f;
}

__device__ __forceinline__ void dotProductTile32FP(float *sum, const int A_row, const int B_col, const float sm_A[TILE_SIZE][TILE_SIZE], const float sm_B[TILE_SIZE][TILE_SIZE])
{
    for (int i = 0; i < TILE_SIZE; ++i)
    {
        *sum += sm_A[A_row][i] * sm_B[i][B_col];
    }
}

__global__ void matrixMultImprovedKernel32FP(const float* A, const float* B, float* C, const unsigned int M, const unsigned int N, const unsigned int K) 
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    __shared__ float sm_A[TILE_SIZE][TILE_SIZE];
    __shared__ float sm_B[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (unsigned int tile_idx = 0; tile_idx <  (N + TILE_SIZE - 1) / TILE_SIZE; ++tile_idx)
    {
        // Loading Phase: Load 2D Tiles into shared memory
        loadTileA32FP(A, sm_A, y, M, N, tile_idx);
        loadTileB32FP(B, sm_B, x, N, K, tile_idx);

        __syncthreads();
        
        // Computation Phase: Perform the dot product (A-row x B-col) for the current part of the tile
        // A[row,:] * B[:, col]
        
        dotProductTile32FP(&sum ,threadIdx.y, threadIdx.x, sm_A, sm_B);
        
        __syncthreads();
    }

    if (x >= K || y >= M) return;
    C[y * K + x] = sum;
}


__device__ __forceinline__ void loadTileA64FP(const double *A, double sm[TILE_SIZE][TILE_SIZE], const int y, const unsigned int M, const unsigned int N, const unsigned int tile_idx)
{
    const unsigned int col_A = tile_idx * TILE_SIZE + threadIdx.x;
    if (y < M && col_A < N)
        sm[threadIdx.y][threadIdx.x] = A[y * N + col_A];
    else
        sm[threadIdx.y][threadIdx.x] = 0.0f;
}

__device__ __forceinline__ void loadTileB64FP(const double *B, double sm[TILE_SIZE][TILE_SIZE], const int x, const int N, const int K, const int tile_idx)
{
    const unsigned int row_B = tile_idx * TILE_SIZE + threadIdx.y;
    if (row_B < N && x < K)
        sm[threadIdx.y][threadIdx.x] = B[row_B * K + x];
    else
        sm[threadIdx.y][threadIdx.x] = 0.0f;
}

__device__ __forceinline__ void dotProductTile64FP(double *sum, const int A_row, const int B_col, const double sm_A[TILE_SIZE][TILE_SIZE], const double sm_B[TILE_SIZE][TILE_SIZE])
{
    for (int i = 0; i < TILE_SIZE; ++i)
    {
        *sum += sm_A[A_row][i] * sm_B[i][B_col];
    }
}

__global__ void matrixMultImprovedKernel64FP(const double* A, const double* B, double* C, const unsigned int M, const unsigned int N, const unsigned int K) 
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    __shared__ double sm_A[TILE_SIZE][TILE_SIZE];
    __shared__ double sm_B[TILE_SIZE][TILE_SIZE];

    double sum = 0.0f;

    for (unsigned int tile_idx = 0; tile_idx <  (N + TILE_SIZE - 1) / TILE_SIZE; ++tile_idx)
    {
        // Loading Phase: Load 2D Tiles into shared memory
        loadTileA64FP(A, sm_A, y, M, N, tile_idx);
        loadTileB64FP(B, sm_B, x, N, K, tile_idx);

        __syncthreads();
        
        // Computation Phase: Perform the dot product (A-row x B-col) for the current part of the tile
        // A[row,:] * B[:, col]
        
        dotProductTile64FP(&sum ,threadIdx.y, threadIdx.x, sm_A, sm_B);
        
        __syncthreads();
    }

    if (x >= K || y >= M) return;
    C[y * K + x] = sum;
}
