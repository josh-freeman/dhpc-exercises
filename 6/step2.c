/*
 * Step 2: Basic Put Operation with Fence Synchronization
 *
 * Implement a ring communication pattern using MPI_Put.
 * Each rank sends its rank number to the next rank in a ring.
 *
 * Before: rank 0 has 0, rank 1 has 1, rank 2 has 2, rank 3 has 3
 * After:  rank 0 has 3, rank 1 has 0, rank 2 has 1, rank 3 has 2
 *
 * Tasks:
 * 1. Create a window exposing your local integer
 * 2. Use MPI_Win_fence to open an epoch
 * 3. Use MPI_Put to send your rank to the next process
 * 4. Use MPI_Win_fence to close the epoch
 * 5. Print the received value
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int *data;
    MPI_Win win;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Allocate and initialize
    MPI_Alloc_mem(sizeof(int), MPI_INFO_NULL, &data);
    *data = rank; // Initially holds own rank

    // Create window
    MPI_Win_create(data, sizeof(int), sizeof(int),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // Calculate next rank in ring
    int next_rank = (rank + 1) % size;

    // TODO: Open an epoch with MPI_Win_fence
    // Signature: MPI_Win_fence(assert, win)
    // Use 0 for assert


    // TODO: Put your rank value into the next process's window
    // Signature: MPI_Put(origin_addr, origin_count, origin_datatype,
    //                    target_rank, target_disp, target_count, target_datatype, win)


    // TODO: Close the epoch with MPI_Win_fence
    // After this fence, the Put is guaranteed complete

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
