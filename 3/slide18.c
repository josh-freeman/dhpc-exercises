#include <stdio.h>

int main()
{
    int number;

#pragma omp parallel
#pragma omp single nowait
    {
#pragma omp task depend(out : number)
        number = 1;

#pragma omp task depend(inout : number)
        {
            printf("I think the number is %d\n", number);
            number++;
        }

#pragma omp task depend(in : number)
        printf("I think the final number is %d\n", number);
    }

    return 0;
}
