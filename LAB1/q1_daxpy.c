#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N (1<<16)

int main() {
    double *X = malloc(N*sizeof(double));
    double *Y = malloc(N*sizeof(double));
    double a = 2.5;
    double start, end;

    for (int i = 0; i < N; i++) {
        X[i] = 1.0;
        Y[i] = 2.0;
    }

    start = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
        X[i] = a * X[i] + Y[i];
    end = omp_get_wtime();

    printf("Time taken: %f seconds\n", end - start);

    free(X); free(Y);
    return 0;
}
