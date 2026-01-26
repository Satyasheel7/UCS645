#include <stdio.h>
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        #pragma omp critical
        {
            printf("Thread %d in critical section\n",
                   omp_get_thread_num());
        }
    }
    return 0;
}
