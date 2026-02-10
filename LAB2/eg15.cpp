// Experiment: Demonstrating False Sharing and Its Mitigation in C++ with OpenMP
// Description: This code illustrates the performance impact of false sharing
// and how padding can be used to mitigate it.
// False sharing occurs when threads on different processors modify variables
// that reside on the same cache line, leading to performance degradation.
// This example compares performance with and without padding.
// C++ Code: False Sharing Example with Padding
#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

const int NUM_THREADS = 8;
const int ITERATIONS = 100000000;
const int CP_SIZE = 64 / sizeof(int); // Number of ints in a 64-byte cache line

int main() {
    // 1. UNPADDED: Shared cache lines
    int partial_sum_unpadded[NUM_THREADS];
    
    // 2. PADDED: Each thread's sum is separated by a full cache line
    int partial_sum_padded[NUM_THREADS][CP_SIZE];

    // Test Unpadded
    auto s1 = high_resolution_clock::now();
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int tid = omp_get_thread_num();
        for(int i=0; i<ITERATIONS; i++) partial_sum_unpadded[tid]++;
    }
    auto e1 = high_resolution_clock::now();

    // Test Padded
    auto s2 = high_resolution_clock::now();
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int tid = omp_get_thread_num();
        for(int i=0; i<ITERATIONS; i++) partial_sum_padded[tid][0]++;
    }
    auto e2 = high_resolution_clock::now();

    cout << "Unpadded Time: " << duration<double>(e1 - s1).count() << "s" << endl;
    cout << "Padded Time:   " << duration<double>(e2 - s2).count() << "s" << endl;

    return 0;
}