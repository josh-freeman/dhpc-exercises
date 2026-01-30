/*
 * Cache Size Detector
 *
 * This program detects CPU cache sizes by measuring memory access latency
 * at different working set sizes. When the working set exceeds a cache level,
 * latency increases noticeably.
 *
 * Technique: Pointer chasing with random access pattern to defeat prefetching.
 *
 * Usage:
 *   gcc -O2 -o cache_size_detector cache_size_detector.c
 *   ./cache_size_detector > cache_results.csv
 *   python3 plot_cache.py
 *
 * Verify results with:
 *   lscpu | grep -i cache
 *   hwloc-ls --only cache
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

// Configuration
#define MIN_SIZE_KB 1          // Start at 1 KB
#define MAX_SIZE_KB (64*1024)  // End at 64 MB
#define ITERATIONS 10000000    // Number of pointer chases per measurement
#define WARMUP_ITERATIONS 1000000  // Warmup iterations
#define NUM_TRIALS 3           // Number of trials to average

/*
 * Setup a pointer-chasing pattern through the array.
 * Uses Fisher-Yates shuffle to create a random cyclic permutation.
 * Each element points to the next element in a random order.
 */
void setup_pointer_chase(size_t *array, size_t n)
{
    // Create array of indices
    size_t *indices = malloc(n * sizeof(size_t));
    if (!indices) {
        fprintf(stderr, "Failed to allocate indices array\n");
        exit(1);
    }

    for (size_t i = 0; i < n; i++) {
        indices[i] = i;
    }

    // Fisher-Yates shuffle
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        size_t temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    // Create the pointer chain: each element points to next in shuffled order
    for (size_t i = 0; i < n - 1; i++) {
        array[indices[i]] = indices[i + 1];
    }
    array[indices[n - 1]] = indices[0];  // Complete the cycle

    free(indices);
}

/*
 * Get current time in nanoseconds
 */
static inline uint64_t get_time_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/*
 * Measure average memory access latency using pointer chasing.
 * Returns latency in nanoseconds per access.
 */
double measure_latency(size_t *array, size_t n, size_t iterations)
{
    size_t idx = 0;

    // Warmup - bring data into cache and stabilize
    for (size_t i = 0; i < WARMUP_ITERATIONS; i++) {
        idx = array[idx];
    }

    // Actual measurement
    uint64_t start = get_time_ns();

    for (size_t i = 0; i < iterations; i++) {
        idx = array[idx];
    }

    uint64_t end = get_time_ns();

    // Prevent compiler from optimizing away the loop
    // (use volatile to force the read)
    volatile size_t dummy = idx;
    (void)dummy;

    double elapsed_ns = (double)(end - start);
    return elapsed_ns / iterations;
}

/*
 * Run multiple trials and return the minimum latency.
 * Minimum is used because higher values are due to noise/interference.
 */
double measure_latency_min(size_t *array, size_t n, size_t iterations, int trials)
{
    double min_latency = 1e9;

    for (int t = 0; t < trials; t++) {
        double latency = measure_latency(array, n, iterations);
        if (latency < min_latency) {
            min_latency = latency;
        }
    }

    return min_latency;
}

int main(int argc, char *argv[])
{
    // Seed random number generator
    srand(time(NULL));

    // Print CSV header
    printf("size_kb,size_bytes,num_elements,latency_ns\n");

    // Test various array sizes
    for (size_t size_kb = MIN_SIZE_KB; size_kb <= MAX_SIZE_KB; size_kb *= 2) {
        // Also test intermediate sizes for better resolution
        size_t sizes_to_test[] = {size_kb, size_kb + size_kb/2};
        int num_sizes = (size_kb * 3 / 2 <= MAX_SIZE_KB) ? 2 : 1;

        for (int s = 0; s < num_sizes; s++) {
            size_t current_size_kb = sizes_to_test[s];
            if (current_size_kb > MAX_SIZE_KB) continue;

            size_t size_bytes = current_size_kb * 1024;
            size_t n = size_bytes / sizeof(size_t);

            // Need at least a few elements for meaningful test
            if (n < 16) continue;

            // Allocate and setup array
            size_t *array = malloc(size_bytes);
            if (!array) {
                fprintf(stderr, "Failed to allocate %zu KB\n", current_size_kb);
                continue;
            }

            setup_pointer_chase(array, n);

            // Adjust iterations for large arrays (to keep runtime reasonable)
            size_t iters = ITERATIONS;
            if (current_size_kb > 1024) {
                iters = ITERATIONS / 2;
            }
            if (current_size_kb > 8192) {
                iters = ITERATIONS / 4;
            }
            if (current_size_kb > 32768) {
                iters = ITERATIONS / 8;
            }

            // Measure latency
            double latency = measure_latency_min(array, n, iters, NUM_TRIALS);

            printf("%zu,%zu,%zu,%.2f\n", current_size_kb, size_bytes, n, latency);
            fflush(stdout);

            // Progress indicator to stderr
            fprintf(stderr, "Tested %zu KB: %.2f ns/access\n", current_size_kb, latency);

            free(array);
        }
    }

    fprintf(stderr, "\nDone! Results written to stdout.\n");
    fprintf(stderr, "Run 'python3 plot_cache.py' to visualize.\n");
    fprintf(stderr, "\nVerify with:\n");
    fprintf(stderr, "  lscpu | grep -i cache\n");
    fprintf(stderr, "  hwloc-ls --only cache\n");

    return 0;
}
