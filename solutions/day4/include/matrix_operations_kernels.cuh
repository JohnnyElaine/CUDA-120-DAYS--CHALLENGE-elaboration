#ifndef MATRIX_OPERATIONS_KERNELS_H
#define MATRIX_OPERATIONS_KERNELS_H

#define TILE_SIZE 16

#include <cuda_runtime.h>

__global__ void matrixAddKernel(const float *A, const float *B, float *C, const unsigned int height, const unsigned int width);
__global__ void matrixMultKernel32FP(const float* A, const float* B, float* C, const unsigned int M, const unsigned int N, const unsigned int K);
__global__ void matrixMultImprovedKernel32FP(const float* A, const float* B, float* C, const unsigned int M, const unsigned int N, const unsigned int K) ;
__global__ void matrixMultImprovedKernel64FP(const double* A, const double* B, double* C, const unsigned int M, const unsigned int N, const unsigned int K) ;

#endif


