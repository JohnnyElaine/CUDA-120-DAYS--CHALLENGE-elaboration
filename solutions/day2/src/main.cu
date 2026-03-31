#include <cuda_runtime.h>
#include "kernels.cuh"

int main()
{
    myKernel<<<3, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}