#include <stdio.h>
#include <omp.h>

int main() {
    double start, end;
    long sum = 0;

    start = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < 100000000; i++)
        sum += i;
    end = omp_get_wtime();

    printf("Time taken: %f seconds\n", end - start);
    return 0;
}
