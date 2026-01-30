/*
 * Step 1: MPI RMA Window Creation
 *
 * Learn how to create an RMA window - the foundation of one-sided communication.
 *
 * Tasks:
 * 1. Allocate memory using MPI_Alloc_mem (better for RMA than malloc)
 * 2. Initialize the memory with your rank value
 * 3. Create an RMA window with MPI_Win_create
 * 4. Print your local value
 * 5. Clean up properly
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

    // TODO: Allocate memory using MPI_Alloc_mem
    // Signature: MPI_Alloc_mem(size_in_bytes, MPI_INFO_NULL, &pointer)
    // Allocate space for one integer


    // TODO: Initialize your data with your rank


    // TODO: Create an RMA window
    // Signature: MPI_Win_create(base, size, disp_unit, info, comm, &win)
    // - base: pointer to memory (data)
    // - size: size in bytes (sizeof(int))
    // - disp_unit: unit for displacement calculations (sizeof(int))
    // - info: MPI_INFO_NULL
    // - comm: MPI_COMM_WORLD
    // - win: output window handle

    // Print local value
    printf("Rank %d: local value = %d\n", rank, *data);

    // TODO: Free the window with MPI_Win_free(&win)

    // TODO: Free the memory with MPI_Free_mem(data)

    MPI_Finalize();
    return 0;
}
