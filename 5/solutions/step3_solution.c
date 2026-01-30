#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int count = 1000;  // Elements per rank

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *sendbuf = malloc(count * sizeof(double));
    double *recvbuf = malloc(count * size * sizeof(double));

    // Initialize sendbuf with some data
    for (int i = 0; i < count; i++) {
        sendbuf[i] = rank * 1000.0 + i;
    }

    // Basic timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Allgather(sendbuf, count, MPI_DOUBLE,
                  recvbuf, count, MPI_DOUBLE, MPI_COMM_WORLD);

    double end = MPI_Wtime();
    double local_time = end - start;


    if (rank == 0){
        double global_time;
        MPI_Reduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        printf("Global time: %.6f seconds\n", global_time);
    }else{
        MPI_Reduce(&local_time, NULL, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }

    free(sendbuf);
    free(recvbuf);
    MPI_Finalize();
    return 0;
}
