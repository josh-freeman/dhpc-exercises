/*
 * Step 1 Solution: MPI RMA Window Creation
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int *data;
    MPI_Win win;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Allocate memory using MPI_Alloc_mem
    MPI_Alloc_mem(sizeof(int), MPI_INFO_NULL, &data);

    // Initialize with rank
    *data = rank;

    // Create RMA window
    MPI_Win_create(data, sizeof(int), sizeof(int),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // Print local value
    printf("Rank %d: local value = %d\n", rank, *data);

    // Free window
    MPI_Win_free(&win);

    // Free memory
    MPI_Free_mem(data);

    MPI_Finalize();
    return 0;
}
