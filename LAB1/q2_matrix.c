#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 500

int main() {
    double *A = malloc(N*N*sizeof(double));
    double *B = malloc(N*N*sizeof(double));
    double *C = malloc(N*N*sizeof(double));
    double start, end;

    for (int i=0;i<N*N;i++) A[i]=B[i]=1.0;

    // 1D threading
    start = omp_get_wtime();
    #pragma omp parallel for
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++) {
            C[i*N+j]=0;
            for (int k=0;k<N;k++)
                C[i*N+j]+=A[i*N+k]*B[k*N+j];
        }
    end = omp_get_wtime();
    printf("1D Parallel Time: %f\n", end-start);

    // 2D threading
    start = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++) {
            C[i*N+j]=0;
            for (int k=0;k<N;k++)
                C[i*N+j]+=A[i*N+k]*B[k*N+j];
        }
    end = omp_get_wtime();
    printf("2D Parallel Time: %f\n", end-start);

    free(A); free(B); free(C);
    return 0;
}
