// Experiment: Demonstrating False Sharing and Its Mitigation in C++ with OpenMP
// Description: This code illustrates the performance impact of false sharing
// and how padding can be used to mitigate it.
// False sharing occurs when threads on different processors modify variables
// that reside on the same cache line, leading to performance degradation.
// This example compares performance with and without padding.
#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

// Standard cache line size is 64 bytes (8 doubles)
const int CACHE_LINE_SIZE = 8; 

struct UnpaddedData {
    double value;
};

struct PaddedData {
    double value;
    double padding[CACHE_LINE_SIZE]; // Force 'value' onto a new cache line
};

int main() {
    const long long iterations = 100000000;
    int n_threads = omp_get_max_threads();
    
    // Arrays where each thread gets its own element
    vector<UnpaddedData> unpadded(n_threads, {0.0});
    vector<PaddedData> padded(n_threads, {0.0});

    cout << left << setw(20) << "Configuration" << "Time (s)" << endl;
    cout << string(40, '-') << endl;

    // --- 1. Testing False Sharing (Unpadded) ---
    auto s1 = high_resolution_clock::now();
    #pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        for (long long i = 0; i < iterations; i++) {
            unpadded[tid].value += 1.0;
        }
    }
    auto e1 = high_resolution_clock::now();
    cout << setw(20) << "False Sharing:" << duration<double>(e1 - s1).count() << "s" << endl;

    // --- 2. Testing No False Sharing (Padded) ---
    auto s2 = high_resolution_clock::now();
    #pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        for (long long i = 0; i < iterations; i++) {
            padded[tid].value += 1.0;
        }
    }
    auto e2 = high_resolution_clock::now();
    cout << setw(20) << "Padded (Fixed):" << duration<double>(e2 - s2).count() << "s" << endl;

    return 0;
}