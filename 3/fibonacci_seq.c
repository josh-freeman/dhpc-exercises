#include <stdio.h>
#include <stdlib.h>

#define N 1000

int main()
{
    long int fib_numbers[N];

    // TODO: Compute the first N Fibonacci numbers sequentially
    // fib_numbers[0] = 0
    // fib_numbers[1] = 1
    // fib_numbers[i] = fib_numbers[i-1] + fib_numbers[i-2]

    printf("The Fibonacci numbers are:");
    for (int i = 0; i < N; i++)
        printf(" %ld", fib_numbers[i]);
    printf("\n");

    return 0;
}
