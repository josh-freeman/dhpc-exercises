#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define WARMUP 5
#define ITERATIONS 100

static void ring_once(int rank, int size, int next, int prev,
                      char *sendbuf, char *recvbuf, int nbytes, MPI_Status *status)
{
    if (rank == 0)
    {
        MPI_Send(sendbuf, nbytes, MPI_BYTE, next, 0, MPI_COMM_WORLD);
        MPI_Recv(recvbuf, nbytes, MPI_BYTE, prev, 0, MPI_COMM_WORLD, status);
    }
    else
    {
        MPI_Recv(recvbuf, nbytes, MPI_BYTE, prev, 0, MPI_COMM_WORLD, status);
        MPI_Send(recvbuf, nbytes, MPI_BYTE, next, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    int sizes[] = {0, 1, 8, 64, 512, 4096, 32768, 262144, 1048576};
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);

    int max_bytes = sizes[nsizes - 1];
    char *sendbuf = calloc(max_bytes, 1);
    char *recvbuf = malloc(max_bytes);
    MPI_Status status;

    if (rank == 0)
        printf("# bytes    time_us\n");

    for (int s = 0; s < nsizes; s++)
    {
        int nbytes = sizes[s];

        for (int w = 0; w < WARMUP; w++)
            ring_once(rank, size, next, prev, sendbuf, recvbuf, nbytes, &status);

        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        for (int it = 0; it < ITERATIONS; it++)
            ring_once(rank, size, next, prev, sendbuf, recvbuf, nbytes, &status);

        double end = MPI_Wtime();
        double avg_us = (end - start) / ITERATIONS * 1e6;

        if (rank == 0)
            printf("%d    %.2f us\n", nbytes, avg_us);
    }

    free(sendbuf);
    free(recvbuf);
    MPI_Finalize();
    return 0;
}
