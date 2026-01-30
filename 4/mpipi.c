#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

static long num_steps = 100000000;
double step;

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int i;
    double x, pi, sum = 0.0;
    step = 1.0 / (double)num_steps;

    for (i = rank*(num_steps/size); i < (rank == size-1? num_steps: (rank+1)*(num_steps/size)); i++)
    {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    MPI_Send(&sum, 1, MPI_DOUBLE, 0, 99, MPI_COMM_WORLD);
    sum=0;
    
    if (rank==0){
        double tmp;

        for (int i=0; i<size; i++){
            MPI_Recv(&tmp,1,MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG, MPI_COMM_WORLD,&status);
            sum+=tmp;
        }


        pi = step * sum; 

        printf("{\"pi\": %lf}\n", pi);
    }


    

    MPI_Finalize();
    return 0;
}