#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define MSG_SIZE 1024
#define WARMUP 5
#define ITERATIONS 100

// TODO: Implement ring_once - a helper that performs one ring communication
// Signature: static void ring_once(int rank, int size, int *sendbuf, int *recvbuf, MPI_Status *status)
// Rank 0 sends then receives; all others receive then send.


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

    // TODO: Warmup phase - call ring_once WARMUP times

    // TODO: Benchmark phase
    // 1. Synchronize with MPI_Barrier
    // 2. Record start time with MPI_Wtime
    // 3. Run ring_once ITERATIONS times
    // 4. Record end time
    // 5. Calculate and print average latency in microseconds (rank 0 only)


    MPI_Finalize();
    return 0;
}
