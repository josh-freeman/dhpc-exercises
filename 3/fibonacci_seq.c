#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv)
{
    int N = 30;
    if (argc > 1)
        N = atoi(argv[1]);

    long long *fib_numbers = malloc(N * sizeof(long long));
    double start_time, run_time;

    start_time = omp_get_wtime();

    fib_numbers[0] = 0;
    fib_numbers[1] = 1;

    for (int i = 2; i < N; i++)
    {
        fib_numbers[i] = fib_numbers[i - 1] + fib_numbers[i - 2];
    }

    run_time = omp_get_wtime() - start_time;

    printf("{\"N\": %d, \"run_time\": %.9f}\n", N, run_time);

    free(fib_numbers);
    return 0;
}
