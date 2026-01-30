#include <stdio.h>
#include <stdlib.h>

#define cudaCheckError(ans)                    \
    {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

#define N (1 << 24)
#define THREADS_PER_BLOCK 256

// ============================================================
// TODO: Implement the divergence-free reduction kernel.
//
// Signature:
//   __global__ void reduce1(int *g_idata, int *g_odata, unsigned int n)
//
// Same structure as reduce0 (shared memory load, tree reduction,
// thread 0 writes output). The ONLY change is the reduction loop
// body: instead of using modular arithmetic to pick active threads,
// compute a contiguous index so that the first N/2 threads are
// active, then N/4, etc. This keeps full warps either all-active
// or all-idle, avoiding divergence.
// ============================================================

__global__ void reduce1(int *g_idata, int *g_odata, unsigned int n)
{
    __shared__ int sdata[THREADS_PER_BLOCK];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    sdata[tid] = i < n ? g_idata[i] : 0;
    __syncthreads();
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        int index = 2 * s * tid;
        if (index < blockDim.x)
            sdata[index] += sdata[index + s];
        __syncthreads();
    }
    g_odata[blockIdx.x] = sdata[0];
}

int main()
{
    size_t bytes = N * sizeof(int);

    int *h_idata = (int *)malloc(bytes);
    for (int i = 0; i < N; i++)
        h_idata[i] = 1;

    int *d_idata, *d_odata;
    // ============================================================
    // TODO: Compute numBlocks from N and THREADS_PER_BLOCK.
    // ============================================================
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaCheckError(cudaMalloc(&d_idata, bytes));
    cudaCheckError(cudaMalloc(&d_odata, numBlocks * sizeof(int)));
    cudaCheckError(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

    // ============================================================
    // TODO: Launch reduce1 with the right grid/block dimensions.
    // ============================================================

    // Warmup
    reduce1<<<numBlocks, THREADS_PER_BLOCK>>>(d_idata, d_odata, N);
    cudaCheckError(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    cudaCheckError(cudaEventCreate(&start));
    cudaCheckError(cudaEventCreate(&stop));

    int nIter = 100;
    cudaCheckError(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++)
    {
        reduce1<<<numBlocks, THREADS_PER_BLOCK>>>(d_idata, d_odata, N);
    }
    cudaCheckError(cudaEventRecord(stop));
    cudaCheckError(cudaEventSynchronize(stop));

    float ms = 0;
    cudaCheckError(cudaEventElapsedTime(&ms, start, stop));
    ms /= nIter;

    int *h_odata = (int *)malloc(numBlocks * sizeof(int));
    cudaCheckError(cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(int),
                              cudaMemcpyDeviceToHost));
    long long sum = 0;
    for (int i = 0; i < numBlocks; i++)
        sum += h_odata[i];

    printf("Sum = %lld (expected %d)\n", sum, N);
    printf("Time: %.3f ms\n", ms);
    printf("Throughput: %.2f GB/s\n", (double)bytes / ms / 1e6);
    if (sum == N)
        printf("PASSED\n");
    else
        printf("FAILED\n");

    cudaCheckError(cudaEventDestroy(start));
    cudaCheckError(cudaEventDestroy(stop));
    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);
    return 0;
}
