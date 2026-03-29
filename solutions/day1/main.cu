#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloWorld() 
{
    printf("Hello World from the Graphics Processing Unit!\n");
}

int main()
{
    helloWorld<<<1, 1>>>();

    cudaDeviceSynchronize();

    printf("Hello Word from the Central Processing Unit!\n");

    return 0;
}