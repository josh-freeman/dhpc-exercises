#include <stdio.h>
#include <stdlib.h>

#define N 1000

int main()
{
    long int fib_numbers[N];
    // TODO: parallelize using tasks (need to find the in and out task dependency)
    // #pragma omp parallel
    // #pragma omp single nowait
    {
        // #pragma omp task depend(out : fib_numbers[0])
        fib_numbers[0] = 0;
        // #pragma omp task depend(out : fib_numbers[1])
        fib_numbers[1] = 1;

        for (int i = 2; i < N; i++)
        {
            // #pragma omp task depend(in : fib_numbers[i - 1], fib_numbers[i - 2]) depend(out : fib_numbers[i])
            fib_numbers[i] = fib_numbers[i - 1] + fib_numbers[i - 2];
        }

        printf("The Fibonacci numbers are:");
        for (int i = 0; i < N; i++)
            // #pragma omp task depend(in : fib_numbers[i])
            printf(" %ld", fib_numbers[i]);
        printf("\n");
    }
    return 0;
}