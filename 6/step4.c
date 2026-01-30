/*
 * Step 4: 2D Heat Diffusion with RMA Halo Exchange
 *
 * This is the main exercise. Implement a 2D heat diffusion simulation
 * using MPI RMA for halo exchange.
 *
 * The simulation uses a 2D Cartesian grid decomposition.
 * Each process owns a local block of the grid with ghost cells.
 *
 * Key concepts:
 * 1. 2D Cartesian topology (MPI_Cart_create)
 * 2. RMA window for the local grid
 * 3. Halo exchange using MPI_Put with Fence synchronization
 * 4. 5-point stencil for heat diffusion
 *
 * Memory layout (row-major):
 *   grid[row * (local_ny + 2) + col]
 *
 * Grid structure (local_nx=4, local_ny=4):
 *   G G G G G G   <- top ghost row
 *   G . . . . G   <- interior rows
 *   G . . . . G
 *   G . . . . G
 *   G . . . . G
 *   G G G G G G   <- bottom ghost row
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

// Grid dimensions (global)
#define NX 100
#define NY 100

// Simulation parameters
#define ALPHA 0.01      // Thermal diffusivity
#define DX 0.1          // Grid spacing
#define DT 0.0001       // Time step (must satisfy CFL condition)
#define NSTEPS 10000000 // Number of time steps
#define OUTPUT_FREQ 100 // Output frequency

// Macro for 2D indexing in row-major order
#define IDX(row, col, ncols) ((row) * (ncols) + (col))

int main(int argc, char *argv[])
{
    int rank, size;
    int dims[2] = {0, 0};    // Let MPI choose dimensions
    int periods[2] = {0, 0}; // Non-periodic boundaries
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

    if (rank == 0)
    {
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

    // Allocate grids (current and next timestep)
    double *grid, *grid_new;
    MPI_Alloc_mem(grid_size * sizeof(double), MPI_INFO_NULL, &grid);
    grid_new = (double *)malloc(grid_size * sizeof(double));

    // Initialize grid to zero with hot spot in center
    memset(grid, 0, grid_size * sizeof(double));
    memset(grid_new, 0, grid_size * sizeof(double));

    // Calculate global position of this process's local grid
    int global_start_x = coords[0] * local_nx;
    int global_start_y = coords[1] * local_ny;

    // Set initial hot spot in the center of the global grid
    int center_x = NX / 2;
    int center_y = NY / 2;

    for (int i = 1; i <= local_nx; i++)
    {
        for (int j = 1; j <= local_ny; j++)
        {
            int gx = global_start_x + (i - 1);
            int gy = global_start_y + (j - 1);

            // Hot spot: circle of radius 10 in center
            double dist = sqrt((gx - center_x) * (gx - center_x) +
                               (gy - center_y) * (gy - center_y));
            if (dist < 10)
            {
                grid[IDX(i, j, grid_cols)] = 100.0;
            }
        }
    }

    // Create RMA window for the grid
    MPI_Win_create(grid, grid_size * sizeof(double), sizeof(double),
                   MPI_INFO_NULL, cart_comm, &win);

    // Get neighbor ranks
    int up, down, left, right;
    MPI_Cart_shift(cart_comm, 0, 1, &up, &down);    // Vertical neighbors
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right); // Horizontal neighbors

    // TODO: Create MPI_Datatype for column exchange (non-contiguous)
    // A column has 'local_nx' elements, each separated by 'grid_cols' elements
    MPI_Datatype col_type;


    // Time stepping loop
    double start_time = MPI_Wtime();

    for (int step = 0; step < NSTEPS; step++)
    {

        // =============================================
        // HALO EXCHANGE using RMA
        // =============================================

        // TODO: Implement halo exchange using MPI_Put with lock/unlock
        // For each neighbor (up, down, left, right):
        //   1. Lock the neighbor's window (MPI_LOCK_SHARED)
        //   2. Put boundary data into the neighbor's ghost cells
        //   3. Unlock the neighbor's window
        //
        // Row exchange: Put local_ny doubles (contiguous row)
        //   - Top row (row 1) -> up neighbor's bottom ghost (row local_nx+1)
        //   - Bottom row (row local_nx) -> down neighbor's top ghost (row 0)
        //
        // Column exchange: Use col_type (non-contiguous column)
        //   - Left column (col 1) -> left neighbor's right ghost (col local_ny+1)
        //   - Right column (col local_ny) -> right neighbor's left ghost (col 0)


        // =============================================
        // COMPUTE: 5-point stencil
        // =============================================

        // TODO: Apply the 5-point stencil heat equation to interior points
        // Formula: new = old + alpha * dt / (dx*dx) * (neighbors - 4*center)
        // Loop over interior points (i=1..local_nx, j=1..local_ny)

        // Swap grids
        double *temp = grid;
        grid = grid_new;
        grid_new = temp;

        // Output progress
        if (step % OUTPUT_FREQ == 0 && rank == 0)
        {
            printf("Step %d / %d\n", step, NSTEPS);
        }
    }

    double end_time = MPI_Wtime();

    // Calculate local statistics
    double local_sum = 0.0, local_max = 0.0;
    for (int i = 1; i <= local_nx; i++)
    {
        for (int j = 1; j <= local_ny; j++)
        {
            double val = grid[IDX(i, j, grid_cols)];
            local_sum += val;
            if (val > local_max)
                local_max = val;
        }
    }

    // Reduce to get global statistics
    double global_sum, global_max;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);

    if (rank == 0)
    {
        printf("\nSimulation complete!\n");
        printf("Time: %.3f seconds\n", end_time - start_time);
        printf("Average temperature: %.4f\n", global_sum / (NX * NY));
        printf("Maximum temperature: %.4f\n", global_max);
    }

    // Output final grid for plotting (rank 0 gathers and writes)
    // Simplified: each rank writes its portion
    char filename[64];
    sprintf(filename, "heat_output_%d.dat", rank);
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "# Rank %d, coords (%d, %d), local size %d x %d\n",
            rank, coords[0], coords[1], local_nx, local_ny);
    for (int i = 1; i <= local_nx; i++)
    {
        for (int j = 1; j <= local_ny; j++)
        {
            int gx = global_start_x + (i - 1);
            int gy = global_start_y + (j - 1);
            fprintf(fp, "%d %d %.6f\n", gx, gy, grid[IDX(i, j, grid_cols)]);
        }
    }
    fclose(fp);

    // Cleanup
    // TODO: Free col_type when you create it

    MPI_Win_free(&win);
    MPI_Free_mem(grid);
    free(grid_new);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();

    return 0;
}
