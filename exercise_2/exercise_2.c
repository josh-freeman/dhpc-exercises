#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static long num_steps = 100000000;
double step;

int main(int argc, char **argv)
{
    if (argc > 1)
        num_steps = strtol(argv[1], NULL, 10);
    int i;
    double x, pi, sum = 0.0;
    double start_time, run_time;
    step = 1.0 / (double)num_steps;
    start_time = omp_get_wtime();
    for (i = 0; i < num_steps; i++)
    {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    pi = step * sum;
    run_time = omp_get_wtime() - start_time;
    printf("{\"pi\": %lf, \"run_time\": %lf}\n", pi, run_time);
}