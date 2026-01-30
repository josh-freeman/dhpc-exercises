#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
  int myrank, size, sbuf=23, rbuf=32;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
      printf("This program requires at least two MPI processes.\n");
      MPI_Finalize();
      return 1;
  }

  if (myrank == 0) {
    MPI_Send(&sbuf,     /* message buffer */
             1,                 /* one data item */
             MPI_INT,            /* data item is an integer */
             1,               /* destination process rank */
             99,            /* user chosen message tag */
             MPI_COMM_WORLD);   /* default communicator */
  } else {
    MPI_Recv(&rbuf, 1, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("received: %i\n", rbuf);
  }

  MPI_Finalize();
  return 0;
}
