#include <cuda_runtime.h>
#include "kernels.h"

int main()
{
    myKernel<<<3, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}