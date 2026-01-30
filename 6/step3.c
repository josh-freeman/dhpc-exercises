/*
 * Step 3: 1D Halo Exchange with RMA
 *
 * Implement halo exchange for a 1D decomposition using MPI_Put.
 *
 * Each process has an array: [ghost_left | interior values | ghost_right]
 * The interior is filled with the rank number.
 * Ghost cells need to be filled from neighbors.
 *
 * Layout (4 processes, LOCAL_SIZE=5):
 *   Rank 0: [X | 0 0 0 0 0 | X]  -> After: [X | 0 0 0 0 0 | 1]
 *   Rank 1: [X | 1 1 1 1 1 | X]  -> After: [0 | 1 1 1 1 1 | 2]
 *   Rank 2: [X | 2 2 2 2 2 | X]  -> After: [1 | 2 2 2 2 2 | 3]
 *   Rank 3: [X | 3 3 3 3 3 | X]  -> After: [2 | 3 3 3 3 3 | X]
 *
 * X = boundary (unchanged)
 *
 * Strategy (Put-based):
 * - Put my leftmost interior value into left neighbor's right ghost
 * - Put my rightmost interior value into right neighbor's left ghost
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define LOCAL_SIZE 5

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Win win;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Total size including ghost cells
    int total_size = LOCAL_SIZE + 2;

    // Allocate array: [ghost_left, interior..., ghost_right]
    double *data;
    MPI_Alloc_mem(total_size * sizeof(double), MPI_INFO_NULL, &data);

    // Initialize: ghost cells = -1, interior = rank
    data[0] = -1.0;              // left ghost
    data[total_size - 1] = -1.0; // right ghost
    for (int i = 1; i <= LOCAL_SIZE; i++)
    {
        data[i] = (double)rank; // interior
    }

    // Create window exposing the entire array
    MPI_Win_create(data, total_size * sizeof(double), sizeof(double),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // Determine neighbors
    int left_neighbor = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int right_neighbor = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    // TODO: Open epoch with fence


    // TODO: Put my leftmost interior value (data[1]) into left neighbor's right ghost

    // TODO: Put my rightmost interior value (data[LOCAL_SIZE]) into right neighbor's left ghost


    // TODO: Close epoch with fence


    // Verify results
    int errors = 0;

    // Check left ghost (should be left neighbor's rank, or -1 if no left neighbor)
    double expected_left = (rank > 0) ? (double)(rank - 1) : -1.0;
    if (data[0] != expected_left)
    {
        printf("Rank %d: LEFT ghost wrong! Got %.0f, expected %.0f\n",
               rank, data[0], expected_left);
        errors++;
    }

    // Check right ghost (should be right neighbor's rank, or -1 if no right neighbor)
    double expected_right = (rank < size - 1) ? (double)(rank + 1) : -1.0;
    if (data[total_size - 1] != expected_right)
    {
        printf("Rank %d: RIGHT ghost wrong! Got %.0f, expected %.0f\n",
               rank, data[total_size - 1], expected_right);
        errors++;
    }

    if (errors == 0)
    {
        printf("Rank %d: Halo exchange correct! [%.0f | %.0f ... %.0f | %.0f]\n",
               rank, data[0], data[1], data[LOCAL_SIZE], data[total_size - 1]);
    }

    MPI_Win_free(&win);
    MPI_Free_mem(data);
    MPI_Finalize();
    return 0;
}
