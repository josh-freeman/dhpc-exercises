#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define NUM_ITERATIONS 100  // Repeat to get stable timing
#define WARMUP_ITERATIONS 10

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Test different data sizes (in number of doubles)
    int data_sizes[] = {1, 10, 100, 1000, 10000, 100000};
    int num_sizes = sizeof(data_sizes) / sizeof(data_sizes[0]);

    if (rank == 0) {
        printf("# MPI_Allgather Benchmark\n");
        printf("# Processes: %d\n", size);
        printf("# Iterations: %d\n", NUM_ITERATIONS);
        printf("#\n");
        printf("# DataSize(doubles)  Bytes_per_rank  Total_bytes  Time(us)  Bandwidth(MB/s)\n");
    }

    for (int s = 0; s < num_sizes; s++) {
        int count = data_sizes[s];  // Elements per rank

        // Allocate buffers
        double *sendbuf = (double *)malloc(count * sizeof(double));
        double *recvbuf = (double *)malloc(count * size * sizeof(double));

        if (!sendbuf || !recvbuf) {
            fprintf(stderr, "Rank %d: malloc failed for size %d\n", rank, count);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Initialize send buffer with some data
        for (int i = 0; i < count; i++) {
            sendbuf[i] = rank * 1000.0 + i;
        }

        // Warmup runs (don't measure these)
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            MPI_Allgather(sendbuf, count, MPI_DOUBLE,
                          recvbuf, count, MPI_DOUBLE,
                          MPI_COMM_WORLD);
        }

        // Synchronize before timing
        MPI_Barrier(MPI_COMM_WORLD);

        // Start timing
        double start_time = MPI_Wtime();

        for (int i = 0; i < NUM_ITERATIONS; i++) {
            MPI_Allgather(sendbuf, count, MPI_DOUBLE,
                          recvbuf, count, MPI_DOUBLE,
                          MPI_COMM_WORLD);
        }

        // End timing
        double end_time = MPI_Wtime();
        double local_time = (end_time - start_time) / NUM_ITERATIONS;

        // Get the maximum time across all ranks (slowest rank determines collective time)
        double max_time;
        MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        // Print results from rank 0
        if (rank == 0) {
            long bytes_per_rank = count * sizeof(double);
            long total_bytes = bytes_per_rank * size;  // Total data moved
            double time_us = max_time * 1e6;
            double bandwidth = (total_bytes / max_time) / (1024.0 * 1024.0);  // MB/s

            printf("%8d %15ld %12ld %12.2f %12.2f\n",
                   count, bytes_per_rank, total_bytes, time_us, bandwidth);
        }

        free(sendbuf);
        free(recvbuf);
    }

    MPI_Finalize();
    return 0;
}
