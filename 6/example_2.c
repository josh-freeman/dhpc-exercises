#include <stdio.h>
#include <mpi.h>

int main(int argc, char ** argv)
{
    int *a;
    MPI_Win win;
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* collectively create remote accessible memory in a window */
    MPI_Win_allocate(1000*sizeof(int), sizeof(int), MPI_INFO_NULL,
                     MPI_COMM_WORLD, &a, &win);

    /* Array 'a' is now accessible from all processes in
     * MPI_COMM_WORLD */

    /* Initialize some values */
    a[0] = rank * 10;
    a[1] = rank * 10 + 1;

    printf("Rank %d: a[0] = %d, a[1] = %d, window allocated successfully\n",
           rank, a[0], a[1]);

    MPI_Win_free(&win);  /* will also free the buffer memory */
    MPI_Finalize();
    return 0;
}
