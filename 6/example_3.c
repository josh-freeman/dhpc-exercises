#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char ** argv)
{
    int *a;
    MPI_Win win;
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, &win);

    /* create private memory */
    a = (int *) malloc(1000 * sizeof(int));

    /* use private memory like you normally would */
    a[0] = 1; a[1] = 2;

    /* locally declare memory as remotely accessible */
    MPI_Win_attach(win, a, 1000*sizeof(int));

    /* Array 'a' is now accessible from all processes */

    printf("Rank %d: a[0] = %d, a[1] = %d, dynamic window attached successfully\n",
           rank, a[0], a[1]);

    /* undeclare remotely accessible memory */
    MPI_Win_detach(win, a);
    free(a);

    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}
