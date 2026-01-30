#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define MSG_SIZE 1024
#define WARMUP 5
#define ITERATIONS 100

static void ring_once(int rank, int size, int *sendbuf, int *recvbuf, MPI_Status *status)
{
    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    if (rank == 0)
    {
        MPI_Send(sendbuf, MSG_SIZE, MPI_INT, next, 0, MPI_COMM_WORLD);
        MPI_Recv(recvbuf, MSG_SIZE, MPI_INT, prev, 0, MPI_COMM_WORLD, status);
    }
    else
    {
        MPI_Recv(recvbuf, MSG_SIZE, MPI_INT, prev, 0, MPI_COMM_WORLD, status);
        MPI_Send(recvbuf, MSG_SIZE, MPI_INT, next, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sendbuf[MSG_SIZE];
    int recvbuf[MSG_SIZE];
    MPI_Status status;

    for (int i = 0; i < MSG_SIZE; i++)
        sendbuf[i] = i;

    /* Warmup */
    for (int w = 0; w < WARMUP; w++)
        ring_once(rank, size, sendbuf, recvbuf, &status);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int it = 0; it < ITERATIONS; it++)
        ring_once(rank, size, sendbuf, recvbuf, &status);

    double end = MPI_Wtime();
    double avg_us = (end - start) / ITERATIONS * 1e6;

    if (rank == 0)
        printf("Average ring latency: %.2f us\n", avg_us);

    MPI_Finalize();
    return 0;
}
