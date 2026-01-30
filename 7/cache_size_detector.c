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
#define _POSIX_C_SOURCE 199309L
#define min(a, b) ((a) < (b) ? (a) : (b))
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdint.h>

// Configuration
#define MIN_SIZE_KB 1             // Start at 1 KB
#define MAX_SIZE_KB (64 * 1024)   // End at 64 MB
#define ITERATIONS 10000000       // Number of pointer chases per measurement
#define WARMUP_ITERATIONS 1000000 // Warmup iterations
#define NUM_TRIALS 3              // Number of trials to average

/*
 * Setup a pointer-chasing pattern through the array.
 * Uses Fisher-Yates shuffle to create a random cyclic permutation.
 * Each element points to the next element in a random order.
 *
 * After this function, following array[i] repeatedly will visit
 * every element exactly once before returning to the start.
 */
void setup_pointer_chase(size_t *array, size_t n)
{
    // TODO: Allocate a temporary array of indices
    // Hint: size_t *indices = malloc(n * sizeof(size_t));

    // TODO: Initialize indices array with 0, 1, 2, ..., n-1

    // TODO: Fisher-Yates shuffle the indices array
    // For i from n-1 down to 1:
    //   Pick random j in [0, i]
    //   Swap indices[i] and indices[j]

    // TODO: Create the pointer chain
    // For each consecutive pair in the shuffled order,
    // make array[shuffled[i]] point to shuffled[i+1]
    // Make the last element point back to the first (complete the cycle)

    // TODO: Free the temporary indices array

}

/*
 * Get current time in nanoseconds using clock_gettime
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

    // TODO: Warmup phase - chase pointers WARMUP_ITERATIONS times
    // This brings data into cache and stabilizes the system

    // TODO: Record start time using get_time_ns()

    // TODO: Main measurement loop - chase pointers 'iterations' times

    // TODO: Prevent compiler optimization by using volatile

    // TODO: Calculate and return nanoseconds per access

    return 0; // placeholder
}

/*
 * Run multiple trials and return the minimum latency.
 * Minimum is used because higher values are due to noise/interference.
 */
double measure_latency_min(size_t *array, size_t n, size_t iterations, int trials)
{
    double min_latency = 1e9;

    // TODO: Run 'trials' measurements and keep the minimum
    // The minimum is typically the most accurate (less interference)


    return min_latency;
}

int main(int argc, char *argv[])
{
    // Seed random number generator
    srand(time(NULL));

    // Print CSV header
    printf("size_kb,size_bytes,num_elements,latency_ns\n");

    // TODO: Loop through array sizes from MIN_SIZE_KB to MAX_SIZE_KB
    // Double the size each iteration (1KB, 2KB, 4KB, 8KB, ...)
    // Optionally test intermediate sizes for better resolution

    // TODO: Implement the main measurement loop
    // For each size from MIN_SIZE_KB to MAX_SIZE_KB (doubling each time):
    //   1. Calculate size_bytes and number of elements n
    //   2. Skip if n < 16
    //   3. Allocate array and setup pointer chase
    //   4. Adjust iterations for larger sizes
    //   5. Measure latency using measure_latency_min()
    //   6. Print results in CSV format: size_kb, size_bytes, n, latency
    //   7. Free the array


    fprintf(stderr, "\nDone! Results written to stdout.\n");
    fprintf(stderr, "Run 'python3 plot_cache.py' to visualize.\n");
    fprintf(stderr, "\nVerify with:\n");
    fprintf(stderr, "  lscpu | grep -i cache\n");
    fprintf(stderr, "  hwloc-ls --only cache\n");

    return 0;
}
