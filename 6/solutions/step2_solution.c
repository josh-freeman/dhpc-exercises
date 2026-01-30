/*
 * Step 2 Solution: Ring Put with Fence Synchronization
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

    // Allocate and initialize
    MPI_Alloc_mem(sizeof(int), MPI_INFO_NULL, &data);
    *data = rank;

    // Create window
    MPI_Win_create(data, sizeof(int), sizeof(int),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // Calculate next rank in ring
    int next_rank = (rank + 1) % size;

    // Open epoch
    MPI_Win_fence(0, win);

    // Put my rank value into next process's window
    MPI_Put(&rank, 1, MPI_INT, next_rank, 0, 1, MPI_INT, win);

    // Close epoch
    MPI_Win_fence(0, win);

    // Print result
    int expected = (rank - 1 + size) % size;
    printf("Rank %d: received %d (expected %d) - %s\n",
           rank, *data, expected,
           (*data == expected) ? "CORRECT" : "WRONG");

    MPI_Win_free(&win);
    MPI_Free_mem(data);
    MPI_Finalize();
    return 0;
}
