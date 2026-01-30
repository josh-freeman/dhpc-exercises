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

#define N (1 << 24) // 16M elements
#define THREADS_PER_BLOCK 256

// ============================================================
// TODO 1: Implement the baseline reduction kernel.
//
// Signature:
//   __global__ void reduce0(int *g_idata, int *g_odata, unsigned int n)
//
// Steps:
//   1. Declare shared memory array of THREADS_PER_BLOCK ints
//   2. Compute tid and global index i
//   3. Load g_idata[i] into sdata[tid] (0 if out of bounds)
//   4. Barrier
//   5. Tree reduction: stride doubles each step (1,2,4,...),
//      only threads at even multiples of the stride participate
//   6. Thread 0 writes block result to g_odata
// ============================================================

__global__ void reduce0(int *g_idata, int *g_odata, unsigned int n)
{
    __shared__ int sdata[THREADS_PER_BLOCK];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    sdata[tid] = i < n ? g_idata[i] : 0;
    __syncthreads();
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        if (tid % (2 * s) == 0)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    g_odata[blockIdx.x] = sdata[0];
}

int main()
{
    size_t bytes = N * sizeof(int);

    // Allocate and initialize host data
    int *h_idata = (int *)malloc(bytes);
    for (int i = 0; i < N; i++)
        h_idata[i] = 1; // sum should be N

    // ============================================================
    // TODO 2: Compute numBlocks from N and THREADS_PER_BLOCK.
    // ============================================================
    int *d_idata, *d_odata;
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaCheckError(cudaMalloc(&d_idata, bytes));
    cudaCheckError(cudaMalloc(&d_odata, numBlocks * sizeof(int)));
    cudaCheckError(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

    // ============================================================
    // TODO 3: Launch reduce0 with numBlocks and THREADS_PER_BLOCK.
    // ============================================================

    // Warmup
    reduce0<<<numBlocks, THREADS_PER_BLOCK>>>(d_idata, d_odata, N);
    cudaCheckError(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    cudaCheckError(cudaEventCreate(&start));
    cudaCheckError(cudaEventCreate(&stop));

    int nIter = 100;
    cudaCheckError(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++)
    {
        reduce0<<<numBlocks, THREADS_PER_BLOCK>>>(d_idata, d_odata, N);
    }
    cudaCheckError(cudaEventRecord(stop));
    cudaCheckError(cudaEventSynchronize(stop));

    float ms = 0;
    cudaCheckError(cudaEventElapsedTime(&ms, start, stop));
    ms /= nIter;

    // Copy back partial sums and finish on CPU
    int *h_odata = (int *)malloc(numBlocks * sizeof(int));
    cudaCheckError(cudaMemcpy(h_odata, d_odata, numBlocks * sizeof(int),
                              cudaMemcpyDeviceToHost));
    long long sum = 0;
    for (int i = 0; i < numBlocks; i++)
        sum += h_odata[i];

    // Verify
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
