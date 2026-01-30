#include <stdio.h>
#include <omp.h>

int main() {
    int n, m;
    int count = 0;
    
    printf("Threads: %d\n", omp_get_max_threads());
    
    // This WILL show a race - shared m causes skipped iterations
    #pragma omp parallel for  // m is shared - BUG!
    for (n = 0; n < 100; n++) {
        for (m = 0; m < 100; m++) {
            #pragma omp atomic
            count++;
        }
    }
    printf("Count (should be 10000): %d\n", count);
    
    count = 0;
    #pragma omp parallel for private(m)  // FIXED
    for (n = 0; n < 100; n++) {
        for (m = 0; m < 100; m++) {
            #pragma omp atomic
            count++;
        }
    }
    printf("Count with private(m): %d\n", count);
    
    return 0;
}
