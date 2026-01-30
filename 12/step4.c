#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define WARMUP 5
#define ITERATIONS 100

// TODO: Implement ring_once - a helper that performs one ring communication
// with variable message size (uses MPI_BYTE instead of MPI_INT)
// Signature: static void ring_once(int rank, int size, int next, int prev,
//                                  char *sendbuf, char *recvbuf, int nbytes, MPI_Status *status)


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

    // TODO: For each message size in sizes[]:
    //   1. Warmup: call ring_once WARMUP times
    //   2. Synchronize with MPI_Barrier
    //   3. Time ITERATIONS ring_once calls
    //   4. Calculate average latency in microseconds
    //   5. Print "nbytes    avg_us us" (rank 0 only)


    free(sendbuf);
    free(recvbuf);
    MPI_Finalize();
    return 0;
}
