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

#define MSG_SIZE 10

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

    // TODO: Implement ring communication
    // - Rank 0: Send sendbuf to next, then receive from prev into recvbuf
    //   Verify received data matches sent data
    // - All other ranks: Receive from prev into recvbuf, then forward to next
    //
    // Use MPI_Send and MPI_Recv with:
    //   MPI_Send(buf, count, MPI_INT, dest, tag, MPI_COMM_WORLD)
    //   MPI_Recv(buf, count, MPI_INT, source, tag, MPI_COMM_WORLD, &status)

    MPI_Finalize();
    return 0;
}
