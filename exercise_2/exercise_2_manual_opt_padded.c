#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static long num_steps = 100000000;
double step;
#define PAD 8 // cache line size is 64 bytes here

int main(int argc, char **argv)
{
    if (argc > 1)
        num_steps = strtol(argv[1], NULL, 10);

    double start_time, run_time;
    step = 1.0 / (double)num_steps;
    start_time = omp_get_wtime();
    int num_threads = 100;
    double pi = 0.0, sums[num_threads][PAD];
    omp_set_num_threads(num_threads);
#pragma omp parallel
    {
        int i, id = omp_get_thread_num();
        double x;
        for (i = id, sums[id][0] = 0.0; i < num_steps; i += num_threads)
        {

            x = (i + 0.5) * step;
            sums[id][0] += 4.0 / (1.0 + x * x);
        }
    }
    double sum = 0.0;
    for (int i = 0; i < num_threads; i++)
        sum += sums[i][0];
    pi = step * sum;
    run_time = omp_get_wtime() - start_time;
    printf("{\"pi\": %lf, \"run_time\": %lf}\n", pi, run_time);
}