#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define PAR

#define THRESH 30

int fib(int n){
    if (n<2) return n;

    int res1, res2;

    #pragma omp task shared(res1) if(n>THRESH)
    res1 = fib(n-1);

    #pragma omp task shared(res2) if(n>THRESH)
    res2 = fib(n-2);
    
    #pragma omp taskwait
    return res1 + res2;
}

int main(int argc, char **argv)
{
    int N = 30;
    if (argc > 1)
        N = atoi(argv[1]);

    long long *fib_numbers = malloc(N * sizeof(long long));
    double start_time, run_time;

    start_time = omp_get_wtime();

    #ifndef PAR
    {
        fib_numbers[0] = 0;
        fib_numbers[1] = 1;

        for (int i = 2; i < N; i++)
        {

            fib_numbers[i] = fib_numbers[i - 1] + fib_numbers[i - 2];
        }

    }
    #endif

    #ifdef PAR
    #pragma omp parallel
    #pragma omp single
    fib_numbers[N-1] = fib(N-1);
    #endif



    run_time = omp_get_wtime() - start_time;

    printf("{\"N\": %d, \"run_time\": %.9f, \"res\": %lld}\n", N, run_time, fib_numbers[N-1]);

    free(fib_numbers);
    return 0;
}
