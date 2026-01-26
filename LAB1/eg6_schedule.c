#include <stdio.h>
#include <omp.h>

int main() {
    #pragma omp parallel for schedule(dynamic,2)
    for (int i = 0; i < 16; i++)
        printf("Thread %d -> i = %d\n",
               omp_get_thread_num(), i);
    return 0;
}
