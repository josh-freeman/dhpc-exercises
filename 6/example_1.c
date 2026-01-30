#include <stdio.h>
#include <mpi.h>

int main(int argc, char ** argv)
{
    int *a;
    MPI_Win win;
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* create private memory */
    MPI_Alloc_mem(1000*sizeof(int), MPI_INFO_NULL, &a);

    /* use private memory like you normally would */
    a[0] = rank+1; a[1] = rank+2;

    /* collectively declare memory as remotely accessible */
    MPI_Win_create(a, 1000*sizeof(int), sizeof(int),
                   MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    
    /* Array 'a' is now accessible by all processes in
     * MPI_COMM_WORLD */

    printf("Rank %d: a[0] = %d, a[1] = %d, window created successfully\n",
           rank, a[0], a[1]);

    MPI_Win_free(&win);
    MPI_Free_mem(a);
    MPI_Finalize();
    return 0;
}
