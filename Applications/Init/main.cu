#include "../../Helpers/DefaultIncludes.h"

const int DSIZE = 4096; // total computations

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__global__ void cuda_hello()
{
    printf("Hello from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);
}

__global__ void cuda_VectorAdd(float* a, float* b, float* c, int Size)
{
    int idx_here = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx_here < Size)
    {
        c[idx_here] = a[idx_here] + b[idx_here];

        printf("(%i) -> A[%f] + B[%f] = C[%f] \n", idx_here, a[idx_here], b[idx_here], c[idx_here]);
    }
}

void HelloWorld()
{
    cuda_hello<<<2, 2 >>> ();
}

void VectorAddition()
{
    // create host memory
    float* a, * b, * c, * d_a, * d_b, * d_c;

    auto size = DSIZE * sizeof(float);

    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);

    std::vector<std::thread> host_thdPool;

    auto RandFn = [&](float* arr)
        {
            for (int i = 0; i < DSIZE; ++i)
            {
                arr[i] = rand() / (float)RAND_MAX;
            }
        };

    host_thdPool.push_back(std::thread(RandFn, a));
    host_thdPool.push_back(std::thread(RandFn, b));

    // create cuda memmory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaCheckErrors("cudaMalloc failure");

    for (auto& thd : host_thdPool)
    {
        thd.join();
    }

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    cudaCheckErrors("cudaMemcpy failure");

    int blockSize = 256; // CUDA maximum is 1024
    int numBlocks = (DSIZE + blockSize - 1) / blockSize;
    // kernel launch
    cuda_VectorAdd << <numBlocks, blockSize >> > (d_a, d_b, d_c, DSIZE); // <<<grid size, block size>>>

    cudaCheckErrors("Kernel Launch failure");

    // copy device to host memory with results
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaCheckErrors("cudaMemcpy Device to Host Copy failure");

    // free host memory
    free(a);
    free(b);
    free(c);

    // free device/cuda memmory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main()
{
    // Hello World!
    {
        //HelloWorld();
    }
    
    // Vector Add
    {
        //VectorAddition();
    }

    cudaDeviceSynchronize();

    return 0;
}