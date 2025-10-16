#include <stdio.h>
#include <omp.h>

int main() {
    int number = 1;

    printf("=== Demo from Slide 5: Data Sharing ===\n\n");

    printf("Test 1: Using private(number)\n");
    printf("Expected: uninitialized values (likely 0)\n");
    number = 1;
    #pragma omp parallel private(number)
    {
        printf("I think the number is %d.\n", number++);
    }

    printf("\nTest 2: Using firstprivate(number)\n");
    printf("Expected: all threads print 1 (initialized from outer scope)\n");
    number = 1;
    #pragma omp parallel firstprivate(number)
    {
        printf("I think the number is %d.\n", number++);
    }

    return 0;
}
