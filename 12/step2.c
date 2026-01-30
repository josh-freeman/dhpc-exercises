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

    // TODO: Add timing using MPI_Barrier and MPI_Wtime
    // Then implement ring communication (same as step1 but with timing):
    // - Rank 0: Send sendbuf to next, receive from prev, verify data
    // - Other ranks: Receive from prev, forward to next
    // Print elapsed time at the end.

    MPI_Finalize();
    return 0;
}
