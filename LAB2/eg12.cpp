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
    const long long N = 100000000; // 100 Million iterations
    double sum_critical = 0.0;
    double sum_reduction = 0.0;

    cout << left << setw(20) << "Method" << setw(15) << "Time (s)" << "Result" << endl;
    cout << string(50, '-') << endl;

    // --- 1. Critical Section Implementation ---
    auto start1 = high_resolution_clock::now();
    #pragma omp parallel for
    for (long long i = 0; i < N; i++) {
        #pragma omp critical
        {
            sum_critical += 1.0;
        }
    }
    auto end1 = high_resolution_clock::now();
    cout << setw(20) << "Critical Section" << setw(15) << duration<double>(end1 - start1).count() << sum_critical << endl;

    // --- 2. Reduction Clause Implementation ---
    auto start2 = high_resolution_clock::now();
    #pragma omp parallel for reduction(+:sum_reduction)
    for (long long i = 0; i < N; i++) {
        sum_reduction += 1.0;
    }
    auto end2 = high_resolution_clock::now();
    cout << setw(20) << "Reduction" << setw(15) << duration<double>(end2 - start2).count() << sum_reduction << endl;

    return 0;
}