// Experiment: Comparing Critical Section vs. Reduction Clause in OpenMP
// Description: This code compares the performance of using a critical section
// versus using the reduction clause for summing a large number of iterations
// The reduction clause is generally more efficient due to lower synchronization overhead
// C++ Code: Synchronization Methods Comparison
#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <iomanip>
using namespace std;
using namespace std::chrono;
int main() {
    const long long N = 10000000; // Smaller N because atomic is still slower than reduction
    double sum_atomic = 0.0;
    auto start = high_resolution_clock::now();
    #pragma omp parallel for
    for (long long i = 0; i < N; i++) {
        // Atomic tells the hardware to handle this safely without a full software lock
        #pragma omp atomic
        sum_atomic += 1.0;
    }
    auto end = high_resolution_clock::now();
    cout << "Atomic Sum Time: " << duration<double>(end - start).count() << "s" << endl;
    return 0;
}