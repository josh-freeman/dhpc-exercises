#include <stdlib.h>
#include <stdio.h>
#include "../tsc_x86.h"

void init(double* T, int N, int M) {
    for (int n=0; n<N; n++) {
        for (int m=0; m<M; m++) {
            T[n*N+m] = 0.0;
        }
    }
    T[(N/2)*N+(M/2)] = 1000.0;
}

void compute_step(double* T1, double* T2, int N, int M) {
    
    // middle part
    int n, m;
    for (n=1; n<N-1; n++) {
        for (m=1; m<M-1; m++) {
            T2[n*N+m] = (T1[(n-1)*N+m] + T1[(n+1)*N+m] + T1[n*N+m+1] + T1[n*N+m-1] + T1[n*N+m]) / 5.0;
        }
    }
    
    // corner cases (only 2 neighbours and self)
    n=0; m=0;
    T2[n*N+m] = (T1[(n+1)*N+m] + T1[n*N+m+1] + T1[n*N+m]) / 3.0;
    n=0; m=M-1;
    T2[n*N+m] = (T1[(n+1)*N+m] + T1[n*N+m-1] + T1[n*N+m]) / 3.0;
    n=N-1; m=0;
    T2[n*N+m] = (T1[(n-1)*N+m] + T1[n*N+m+1] + T1[n*N+m]) / 3.0;
    n=N-1; m=M-1;
    T2[n*N+m] = (T1[(n-1)*N+m] + T1[n*N+m-1] + T1[n*N+m]) / 3.0;
    
    //edge cases (only 3 neighbours and self)
    n=0;
    for (m=1; m<M-1; m++) T2[n*N+m] = (T1[(n+1)*N+m] + T1[n*N+m+1] + T1[n*N+m-1] + T1[n*N+m]) / 4.0;
    n=N-1;
    for (m=1; m<M-1; m++) T2[n*N+m] = (T1[(n-1)*N+m] + T1[n*N+m+1] + T1[n*N+m-1] + T1[n*N+m]) / 4.0;
    m=0;
    for (n=1; n<N-1; n++) T2[n*N+m] = (T1[(n+1)*N+m] + T1[(n-1)*N+m] + T1[n*N+m+1] + T1[n*N+m]) / 4.0;
    m=M-1;
    for (n=1; n<N-1; n++) T2[n*N+m] = (T1[(n+1)*N+m] + T1[(n-1)*N+m] + T1[n*N+m-1] + T1[n*N+m]) / 4.0;
}

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 100;
    int M = N;
    int num_steps = 100;
    double* T1 = malloc(sizeof(double)*N*M);
    double* T2 = malloc(sizeof(double)*N*M);
    
    init(T1, N, M);
    init_tsc();
    myInt64 tsc = start_tsc();
    for (int t=0; t<num_steps; t++) {
        compute_step(T1, T2, N, M);
        // swap buffers
        double* tmp = T1; T1 = T2; T2 = tmp;
    }
    tsc = stop_tsc(tsc);
    printf("{\"cycles\": %lld}\n", tsc);
    free(T1);
    free(T2);
}