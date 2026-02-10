// Experiment: Comparing Synchronization Methods in C++ with OpenMP
// Description: This code compares the performance of two common synchronization
// methods in OpenMP: critical sections and reduction clauses.
// The critical section method serializes access to a shared variable,
// while the reduction method allows threads to maintain private copies
// and combines them at the end, reducing contention.
// C++ Code: Synchronization Methods Comparison
#include <iostream>
#include <omp.h>
#include <chrono>
#include <iomanip>
using namespace std;
using namespace std::chrono;
void test_sync_methods(long long N) {
    double sum_critical = 0.0;
    double sum_reduction = 0.0;
    // 1. CRITICAL SECTION (The "One-at-a-time" approach)
    // Threads must wait for a software lock to update the global variable.
    auto start = high_resolution_clock::now();
    #pragma omp parallel for
    for (long long i = 0; i < N; i++) {
        #pragma omp critical
        {
            sum_critical += 1.0; 
        }
    }
    auto end = high_resolution_clock::now();
    duration<double> time_critical = end - start;
    // 2. REDUCTION (The "Local-sum" approach)
    // Each thread gets a private copy of the variable and merges at the end.
    start = high_resolution_clock::now();
    #pragma omp parallel for reduction(+:sum_reduction)
    for (long long i = 0; i < N; i++) {
        sum_reduction += 1.0;
    }
    end = high_resolution_clock::now();
    duration<double> time_reduction = end - start;
    // Results
    cout << left << setw(20) << "Method" << "Time (s)" << endl;
    cout << "------------------------------------" << endl;
    cout << setw(20) << "Critical Section" << time_critical.count() << "s" << endl;
    cout << setw(20) << "Reduction" << time_reduction.count() << "s" << endl;    
    if (time_reduction.count() > 0) {
        cout << "\nOverhead Factor: " << (time_critical.count() / time_reduction.count()) 
             << "x slower" << endl;
    }
}
int main() {
    // 10 Million iterations
    test_sync_methods(10000000); 
    return 0;
}