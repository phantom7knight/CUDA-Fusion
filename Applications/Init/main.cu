// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

__global__ void cuda_hello()
{
    printf("Hello World from GPU!\n");
}

int main()
{
    cuda_hello<<<1, 1 >>> ();
    return 0;
}