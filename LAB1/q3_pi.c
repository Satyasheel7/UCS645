#include <stdio.h>
#include <omp.h>

static long num_steps = 1000000;

int main() {
    double step = 1.0 / (double)num_steps;
    double sum = 0.0, pi;
    double start, end;

    start = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < num_steps; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x*x);
    }
    pi = step * sum;
    end = omp_get_wtime();

    printf("Pi = %.10f\n", pi);
    printf("Time = %f seconds\n", end - start);
    return 0;
}
