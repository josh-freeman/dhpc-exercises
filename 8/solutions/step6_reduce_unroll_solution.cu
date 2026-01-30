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
// TODO 1: Write a device function for the last-warp reduction.
//
// Signature:
//   __device__ void warpReduce(int* sdata, int tid)
//
// When fewer than 32 threads remain, they all fit in one warp.
// Since a warp executes in lockstep, you can skip __syncthreads()
// and manually unroll the final 6 reduction steps (offsets 32
// down to 1). Use __syncwarp() after each shared memory write
// to ensure visibility under Independent Thread Scheduling.
// ============================================================
__device__ void warpReduce(int *sdata, int tid)
{
    int v = sdata[tid];
    v += sdata[tid + 32];
    __syncwarp();

    sdata[tid] = v;
    __syncwarp(); // not sure why?

    v += sdata[tid + 16];
    __syncwarp();

    sdata[tid] = v;
    __syncwarp();

    v += sdata[tid + 8];
    __syncwarp();

    sdata[tid] = v;
    __syncwarp();

    v += sdata[tid + 4];
    __syncwarp();

    sdata[tid] = v;
    __syncwarp();

    v += sdata[tid + 2];
    __syncwarp();

    sdata[tid] = v;
    __syncwarp();

    v += sdata[tid + 1];
    __syncwarp();

    sdata[tid] = v;
}
// ============================================================
// TODO 2: Write the main reduction kernel.
//
// Signature:
//   __global__ void reduce4(int *g_idata, int *g_odata, unsigned int n)
//
// Same as reduce3 (first-add-during-load) but stop the
// reduction loop early --- once the stride is small enough
// that all remaining active threads fit in a single warp,
// hand off to your warpReduce function instead of continuing
// the loop with barriers.
// ============================================================
__global__ void reduce2(int *g_idata, int *g_odata, unsigned int n)
{
    __shared__ int sdata[THREADS_PER_BLOCK];
    int i = threadIdx.x + 2 * blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    sdata[tid] = i < n ? g_idata[i] : 0;
    if (i + blockDim.x < n)
        sdata[tid] += g_idata[i + blockDim.x];
    __syncthreads();
    for (int s = blockDim.x / 2; s > 32; s /= 2)
    {

        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32)
        warpReduce(sdata, tid);
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
    // TODO: Compute numBlocks (same stride as step 5).
    // ============================================================
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / 2 / THREADS_PER_BLOCK;

    cudaCheckError(cudaMalloc(&d_idata, bytes));
    cudaCheckError(cudaMalloc(&d_odata, numBlocks * sizeof(int)));
    cudaCheckError(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

    // ============================================================
    // TODO: Launch reduce4 with the right grid/block dimensions.
    // ============================================================

    // Warmup
    reduce2<<<numBlocks, THREADS_PER_BLOCK>>>(d_idata, d_odata, N);
    cudaCheckError(cudaDeviceSynchronize());

    // Timing
    cudaEvent_t start, stop;
    cudaCheckError(cudaEventCreate(&start));
    cudaCheckError(cudaEventCreate(&stop));

    int nIter = 100;
    cudaCheckError(cudaEventRecord(start));
    for (int i = 0; i < nIter; i++)
    {
        reduce2<<<numBlocks, THREADS_PER_BLOCK>>>(d_idata, d_odata, N);
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
