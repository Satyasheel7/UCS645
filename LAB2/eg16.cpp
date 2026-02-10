// Experiment: Measuring Memory Bandwidth with OpenMP in C++
// Description: This code measures the memory bandwidth of a simple vector operation
// (A = B + alpha * C) using OpenMP for parallelization.
// It calculates the total data moved and computes the effective bandwidth in GB/s.
// C++ Code: Memory Bandwidth Measurement
#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
using namespace std;
using namespace std::chrono;
int main() {
    long N = 100000000; // 100 Million elements
    vector<double> A(N, 0.0), B(N, 1.0), C(N, 2.0);
    double alpha = 0.5;
    int threads = omp_get_max_threads();    
    auto start = high_resolution_clock::now();
    #pragma omp parallel for num_threads(threads)
    for (long i = 0; i < N; i++) {
        A[i] = B[i] + alpha * C[i];
    }
    auto end = high_resolution_clock::now();    
    duration<double> elapsed = end - start;    
    // Total Data Moved: 2 Reads (B,C) + 1 Write (A) = 3 * N * 8 bytes
    double total_bytes = 3.0 * N * sizeof(double);
    double bandwidth_gb_s = (total_bytes / elapsed.count()) / 1e9;
    cout << "Threads: " << threads << endl;
    cout << "Time:    " << elapsed.count() << " s" << endl;
    cout << "Bandwidth: " << bandwidth_gb_s << " GB/s" << endl;
    return 0;
}