#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

// Exercise: Implement a ring communication program.
// Rank 0 sends a message of MSG_SIZE ints around a ring of P processes.
// Each rank receives from its predecessor and forwards to its successor.
// Rank 0 verifies the data comes back intact.
//
// See exercise_guide.pdf for details. Hints available there if needed.

#define MSG_SIZE 1024

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;
    int sendbuf[MSG_SIZE] = {0};
    int recvbuf[MSG_SIZE];

    for (int i = 0; i < MSG_SIZE; i++)
    {
        sendbuf[i] = i;
    }
    MPI_Status status;
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    if (rank == 0)
    {
        MPI_Send(sendbuf, MSG_SIZE, MPI_INT, next, 0, MPI_COMM_WORLD);
        MPI_Recv(recvbuf, MSG_SIZE, MPI_INT, prev, 0, MPI_COMM_WORLD, &status);
        for (int i = 0; i < MSG_SIZE; i++)
        {
            if (recvbuf[i] != sendbuf[i])
            {
                printf("Is not ok.");
                MPI_Finalize();
                return 0;
            }
        }
        printf("ring complete. OK");
    }
    else
    {
        printf("forwarded");
        MPI_Recv(recvbuf, MSG_SIZE, MPI_INT, prev, 0, MPI_COMM_WORLD, &status);
        MPI_Send(recvbuf, MSG_SIZE, MPI_INT, next, 0, MPI_COMM_WORLD);
    }

    double end = MPI_Wtime();
    printf("time : %f seconds\n", end - start);
    MPI_Finalize();
    return 0;
}
