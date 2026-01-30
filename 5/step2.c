#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each rank has one integer to send
    int sendbuf = (rank + 1) * 10;  // 10, 20, 30, 40...

    // Each rank receives 'size' integers
    int *recvbuf = malloc(size * sizeof(int));

    // TODO: Call MPI_Allgather here
    // Signature: MPI_Allgather(sendbuf, sendcount, sendtype,
    //                          recvbuf, recvcount, recvtype, comm)
    //
    // Hints:
    // - sendbuf is an int, so pass &sendbuf
    // - sendcount = 1 (we send 1 element)
    // - sendtype = MPI_INT
    // - recvcount = 1 (receive 1 element FROM EACH rank)
    // - recvtype = MPI_INT
    // - comm = MPI_COMM_WORLD


    // Print results
    printf("Rank %d received: ", rank);
    for (int i = 0; i < size; i++) {
        printf("%d ", recvbuf[i]);
    }
    printf("\n");

    free(recvbuf);
    MPI_Finalize();
    return 0;
}
