#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int dims[2], periods[2] = {0, 0}, coords[2];
    int up, down, left, right;
    MPI_Comm cart_comm;

    int sendbuf;
    int recvbuf_up, recvbuf_down, recvbuf_left, recvbuf_right;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set up a 2D grid - let MPI figure out the dimensions
    dims[0] = 0;  // Number of rows (0 = let MPI decide)
    dims[1] = 0;  // Number of columns (0 = let MPI decide)
    MPI_Dims_create(size, 2, dims);

    if (rank == 0) {
        printf("Creating a %d x %d Cartesian grid with %d processes\n\n",
               dims[0], dims[1], size);
    }

    // Create Cartesian communicator
    // periods = {0,0} means non-periodic (no wrap-around)
    // reorder = 1 means MPI can reorder ranks for efficiency
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    // Get the coordinates of the current process in the grid
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    // Get the ranks of the neighbors (up, down, left, right)
    // MPI_Cart_shift(comm, direction, displacement, &source, &dest)
    // direction 0 = row direction (up/down)
    // direction 1 = column direction (left/right)
    MPI_Cart_shift(cart_comm, 0, 1, &up, &down);    // Vertical shifts
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right); // Horizontal shifts

    // Example data to send: use the rank of the process
    sendbuf = rank;

    // Initialize receive buffers to -1 (indicates no neighbor)
    recvbuf_up = -1;
    recvbuf_down = -1;
    recvbuf_left = -1;
    recvbuf_right = -1;

    // Exchange data with neighbors
    // Send to up, receive from down
    MPI_Sendrecv(&sendbuf, 1, MPI_INT, up, 0,
                 &recvbuf_down, 1, MPI_INT, down, 0,
                 cart_comm, MPI_STATUS_IGNORE);

    // Send to down, receive from up
    MPI_Sendrecv(&sendbuf, 1, MPI_INT, down, 0,
                 &recvbuf_up, 1, MPI_INT, up, 0,
                 cart_comm, MPI_STATUS_IGNORE);

    // Send to left, receive from right
    MPI_Sendrecv(&sendbuf, 1, MPI_INT, left, 0,
                 &recvbuf_right, 1, MPI_INT, right, 0,
                 cart_comm, MPI_STATUS_IGNORE);

    // Send to right, receive from left
    MPI_Sendrecv(&sendbuf, 1, MPI_INT, right, 0,
                 &recvbuf_left, 1, MPI_INT, left, 0,
                 cart_comm, MPI_STATUS_IGNORE);

    // Print results (use barrier to keep output somewhat ordered)
    for (int i = 0; i < size; i++) {
        MPI_Barrier(cart_comm);
        if (rank == i) {
            printf("Rank %d at coords (%d, %d):\n", rank, coords[0], coords[1]);
            printf("  Neighbors: up=%2d, down=%2d, left=%2d, right=%2d\n",
                   up, down, left, right);
            printf("  Received:  up=%2d, down=%2d, left=%2d, right=%2d\n\n",
                   recvbuf_up, recvbuf_down, recvbuf_left, recvbuf_right);
        }
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
