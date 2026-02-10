// example of strong scaling in C++ using OpenMP
// measuring performance of a simple vector operation (A = B + scalar * C)
// as we increase the number of threads
// to illustrate strong scaling behavior
// and memory bandwidth considerations
// Note: Ensure your system has enough memory to handle large arrays
// This is a memory-bound operation because very little math is 
// done compared to the amount of data being moved.
 // C++ Code: Vector Triad (The "Stream" Benchmark)
#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <string>

using namespace std;
using namespace std::chrono;

int main() {
    // Large enough to exceed L3 cache (~800MB per vector, ~2.4GB total)
    long long N = 100000000; 
    vector<double> A(N, 0.0), B(N, 1.1), C(N, 2.2);
    double scalar = 3.3;

    int max_threads = omp_get_max_threads();
    
    cout << "Memory Bandwidth & Scalability Test (Triad Kernel)" << endl;
    cout << left << setw(10) << "Cores" 
              << setw(15) << "Time (s)" 
              << setw(15) << "BW (GB/s)" 
              << "Speedup" << endl;
    cout << string(55, '-') << endl;

    double t_serial = 0;

    for (int threads = 1; threads <= max_threads; threads++) {
        auto start = high_resolution_clock::now();

        // The "Triad" operation: A = B + s*C
        #pragma omp parallel for num_threads(threads)
        for (long long i = 0; i < N; i++) {
            A[i] = B[i] + scalar * C[i];
        }

        auto end = high_resolution_clock::now();
        duration<double> diff = end - start;
        double Tp = diff.count();

        if (threads == 1) t_serial = Tp;

        // Metric: 3 arrays (A, B, C) x N elements x 8 bytes per double
        double total_data_gb = (3.0 * N * sizeof(double)) / 1e9;
        double BW = total_data_gb / Tp;

        cout << left << setw(10) << threads 
                  << setw(15) << Tp 
                  << setw(15) << BW 
                  << fixed << setprecision(2) << (t_serial / Tp) << "x" << endl;
    }

    return 0;
}