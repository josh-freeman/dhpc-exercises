#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Error checking macro
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

// ============================================================
// TODO 1: Write the vector addition kernel.
//
// Signature:
//   __global__ void vectorAdd(const float* a, const float* b, float* c, int n)
//
// Each thread should:
//   - Compute its global thread ID
//   - Check that the ID is within bounds (< n)
//   - Write c[id] = a[id] + b[id]
// ============================================================


int main()
{
    int n = 1000000;
    size_t size = n * sizeof(float);

    // Allocate host vectors
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    // Initialize host vectors
    for (int i = 0; i < n; i++)
    {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    // Device vectors
    float *d_a, *d_b, *d_c;

    // ============================================================
    // TODO 2: Allocate device memory for d_a, d_b, d_c
    //         using cudaMalloc. Each needs 'size' bytes.
    // ============================================================


    // ============================================================
    // TODO 3: Copy h_a -> d_a and h_b -> d_b using cudaMemcpy.
    //         Direction: cudaMemcpyHostToDevice
    // ============================================================


    // Set up grid and block dimensions
    int blockSize = 256;

    // ============================================================
    // TODO 4: Compute numBlocks so that blockSize * numBlocks >= n.
    //         Then launch: vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);
    // ============================================================

    // Check for kernel launch errors
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    // ============================================================
    // TODO 5: Copy d_c -> h_c using cudaMemcpy.
    //         Direction: cudaMemcpyDeviceToHost
    // ============================================================

    // Verify result
    int errors = 0;
    for (int i = 0; i < n; i++)
    {
        if (fabs(h_a[i] + h_b[i] - h_c[i]) > 1e-5)
        {
            errors++;
        }
    }

    if (errors == 0)
    {
        printf("PASSED --- all %d values correct.\n", n);
    }
    else
    {
        printf("FAILED --- %d errors.\n", errors);
    }

    // ============================================================
    // TODO 6: Free device memory with cudaFree (d_a, d_b, d_c).
    // ============================================================

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
