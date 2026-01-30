/*
 * Step 4 Solution: 2D Heat Diffusion with RMA Halo Exchange
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define NX 100
#define NY 100
#define ALPHA 0.01
#define DX 0.1
#define DT 0.0001
#define NSTEPS 1000
#define OUTPUT_FREQ 100

#define IDX(row, col, ncols) ((row) * (ncols) + (col))

int main(int argc, char *argv[]) {
    int rank, size;
    int dims[2] = {0, 0};
    int periods[2] = {0, 0};
    int coords[2];
    MPI_Comm cart_comm;
    MPI_Win win;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Create 2D Cartesian topology
    MPI_Dims_create(size, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    if (rank == 0) {
        printf("2D Heat Diffusion with MPI RMA\n");
        printf("Grid: %d x %d, Processes: %d (%d x %d)\n",
               NX, NY, size, dims[0], dims[1]);
        printf("Steps: %d, Alpha: %.4f, dt: %.6f, dx: %.4f\n",
               NSTEPS, ALPHA, DT, DX);
    }

    // Calculate local grid size
    int local_nx = NX / dims[0];
    int local_ny = NY / dims[1];

    // Include ghost cells
    int grid_rows = local_nx + 2;
    int grid_cols = local_ny + 2;
    int grid_size = grid_rows * grid_cols;

    // Allocate grids
    double *grid, *grid_new;
    MPI_Alloc_mem(grid_size * sizeof(double), MPI_INFO_NULL, &grid);
    grid_new = (double *)malloc(grid_size * sizeof(double));

    // Initialize
    memset(grid, 0, grid_size * sizeof(double));
    memset(grid_new, 0, grid_size * sizeof(double));

    int global_start_x = coords[0] * local_nx;
    int global_start_y = coords[1] * local_ny;

    int center_x = NX / 2;
    int center_y = NY / 2;

    for (int i = 1; i <= local_nx; i++) {
        for (int j = 1; j <= local_ny; j++) {
            int gx = global_start_x + (i - 1);
            int gy = global_start_y + (j - 1);

            double dist = sqrt((gx - center_x) * (gx - center_x) +
                               (gy - center_y) * (gy - center_y));
            if (dist < 10) {
                grid[IDX(i, j, grid_cols)] = 100.0;
            }
        }
    }

    // Create RMA window
    MPI_Win_create(grid, grid_size * sizeof(double), sizeof(double),
                   MPI_INFO_NULL, cart_comm, &win);

    // Get neighbors
    int up, down, left, right;
    MPI_Cart_shift(cart_comm, 0, 1, &up, &down);
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right);

    // Create column datatype for non-contiguous exchange
    MPI_Datatype col_type;
    MPI_Type_vector(local_nx, 1, grid_cols, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    double start_time = MPI_Wtime();

    for (int step = 0; step < NSTEPS; step++) {

        // === HALO EXCHANGE ===
        MPI_Win_fence(0, win);

        // Exchange top/bottom rows
        if (up != MPI_PROC_NULL) {
            MPI_Put(&grid[IDX(1, 1, grid_cols)], local_ny, MPI_DOUBLE,
                    up, IDX(local_nx + 1, 1, grid_cols), local_ny, MPI_DOUBLE, win);
        }

        if (down != MPI_PROC_NULL) {
            MPI_Put(&grid[IDX(local_nx, 1, grid_cols)], local_ny, MPI_DOUBLE,
                    down, IDX(0, 1, grid_cols), local_ny, MPI_DOUBLE, win);
        }

        // Exchange left/right columns
        if (left != MPI_PROC_NULL) {
            MPI_Put(&grid[IDX(1, 1, grid_cols)], 1, col_type,
                    left, IDX(1, local_ny + 1, grid_cols), 1, col_type, win);
        }

        if (right != MPI_PROC_NULL) {
            MPI_Put(&grid[IDX(1, local_ny, grid_cols)], 1, col_type,
                    right, IDX(1, 0, grid_cols), 1, col_type, win);
        }

        MPI_Win_fence(0, win);

        // === COMPUTE: 5-point stencil ===
        for (int i = 1; i <= local_nx; i++) {
            for (int j = 1; j <= local_ny; j++) {
                double center = grid[IDX(i, j, grid_cols)];
                double up_val = grid[IDX(i-1, j, grid_cols)];
                double down_val = grid[IDX(i+1, j, grid_cols)];
                double left_val = grid[IDX(i, j-1, grid_cols)];
                double right_val = grid[IDX(i, j+1, grid_cols)];

                grid_new[IDX(i, j, grid_cols)] = center +
                    ALPHA * DT / (DX * DX) *
                    (up_val + down_val + left_val + right_val - 4.0 * center);
            }
        }

        // Swap grids
        double *temp = grid;
        grid = grid_new;
        grid_new = temp;

        if (step % OUTPUT_FREQ == 0 && rank == 0) {
            printf("Step %d / %d\n", step, NSTEPS);
        }
    }

    double end_time = MPI_Wtime();

    // Statistics
    double local_sum = 0.0, local_max = 0.0;
    for (int i = 1; i <= local_nx; i++) {
        for (int j = 1; j <= local_ny; j++) {
            double val = grid[IDX(i, j, grid_cols)];
            local_sum += val;
            if (val > local_max) local_max = val;
        }
    }

    double global_sum, global_max;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    if (rank == 0) {
        printf("\nSimulation complete!\n");
        printf("Time: %.3f seconds\n", end_time - start_time);
        printf("Average temperature: %.4f\n", global_sum / (NX * NY));
        printf("Maximum temperature: %.4f\n", global_max);
    }

    // Output for plotting
    char filename[64];
    sprintf(filename, "heat_output_%d.dat", rank);
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "# Rank %d, coords (%d, %d), local size %d x %d\n",
            rank, coords[0], coords[1], local_nx, local_ny);
    for (int i = 1; i <= local_nx; i++) {
        for (int j = 1; j <= local_ny; j++) {
            int gx = global_start_x + (i - 1);
            int gy = global_start_y + (j - 1);
            fprintf(fp, "%d %d %.6f\n", gx, gy, grid[IDX(i, j, grid_cols)]);
        }
    }
    fclose(fp);

    // Cleanup
    MPI_Type_free(&col_type);
    MPI_Win_free(&win);
    MPI_Free_mem(grid);
    free(grid_new);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();

    return 0;
}
