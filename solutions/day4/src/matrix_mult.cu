#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"

int main() {
    int width = 1024;
    size_t size = width * width * sizeof(float);

    int arr[2][2] = {{10, 20}, {30, 40}};
    int val;
    val = arr[1][1];
    printf("%d\n", val);
    val = 1[arr][1];
    printf("%d\n", val);


    return 0;
}