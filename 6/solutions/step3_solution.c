/*
 * Step 3 Solution: 1D Halo Exchange with RMA
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define LOCAL_SIZE 5

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Win win;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Total size including ghost cells
    int total_size = LOCAL_SIZE + 2;

    // Allocate array
    double *data;
    MPI_Alloc_mem(total_size * sizeof(double), MPI_INFO_NULL, &data);

    // Initialize
    data[0] = -1.0;
    data[total_size - 1] = -1.0;
    for (int i = 1; i <= LOCAL_SIZE; i++) {
        data[i] = (double)rank;
    }

    // Create window
    MPI_Win_create(data, total_size * sizeof(double), sizeof(double),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    // Determine neighbors
    int left_neighbor = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int right_neighbor = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    // Open epoch
    MPI_Win_fence(0, win);

    // Put leftmost interior to left neighbor's right ghost
    if (left_neighbor != MPI_PROC_NULL) {
        MPI_Put(&data[1], 1, MPI_DOUBLE, left_neighbor,
                LOCAL_SIZE + 1, 1, MPI_DOUBLE, win);
    }

    // Put rightmost interior to right neighbor's left ghost
    if (right_neighbor != MPI_PROC_NULL) {
        MPI_Put(&data[LOCAL_SIZE], 1, MPI_DOUBLE, right_neighbor,
                0, 1, MPI_DOUBLE, win);
    }

    // Close epoch
    MPI_Win_fence(0, win);

    // Verify results
    int errors = 0;

    double expected_left = (rank > 0) ? (double)(rank - 1) : -1.0;
    if (data[0] != expected_left) {
        printf("Rank %d: LEFT ghost wrong! Got %.0f, expected %.0f\n",
               rank, data[0], expected_left);
        errors++;
    }

    double expected_right = (rank < size - 1) ? (double)(rank + 1) : -1.0;
    if (data[total_size - 1] != expected_right) {
        printf("Rank %d: RIGHT ghost wrong! Got %.0f, expected %.0f\n",
               rank, data[total_size - 1], expected_right);
        errors++;
    }

    if (errors == 0) {
        printf("Rank %d: Halo exchange correct! [%.0f | %.0f ... %.0f | %.0f]\n",
               rank, data[0], data[1], data[LOCAL_SIZE], data[total_size - 1]);
    }

    MPI_Win_free(&win);
    MPI_Free_mem(data);
    MPI_Finalize();
    return 0;
}
