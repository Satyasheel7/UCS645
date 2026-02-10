// example of cache tiling/blocking in C++ using OpenMP
// This code demonstrates how to improve cache utilization
// by processing data in smaller blocks that fit into the CPU cache
// This technique can significantly speed up memory-bound operations
// especially on large datasets.
// Note: Adjust BLOCK_SIZE based on your CPU's cache size for optimal performance
// C++ Code: Cache Tiling Example

#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iomanip>

using namespace std;
using namespace std::chrono;

const int N = 8192; 
const int BLOCK_SIZE = 64; // Fits nicely in most L2 caches (64*64*8 bytes = 32KB)

void process_standard(vector<double>& data) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // Simple row-major access
            data[i * N + j] = sqrt(data[i * N + j]) * 1.01;
        }
    }
}

void process_with_tiling(vector<double>& data) {
    // collapse(2) merges the i and j loops into one large iteration space for better load balancing
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            
            // Inner loops process the small "tile"
            for (int ii = i; ii < min(i + BLOCK_SIZE, N); ++ii) {
                for (int jj = j; jj < min(j + BLOCK_SIZE, N); ++jj) {
                    data[ii * N + jj] = sqrt(data[ii * N + jj]) * 1.01;
                }
            }
        }
    }
}

int main() {
    // 8192^2 doubles ≈ 536MB (Fits in RAM, but definitely not in Cache)
    vector<double> data(N * N, 42.0);
    
    cout << "Comparing Standard vs Tiled Processing (" << N << "x" << N << ")" << endl;
    cout << "Block Size: " << BLOCK_SIZE << endl;
    cout << "-------------------------------------------------------" << endl;

    // Test Standard
    auto s1 = high_resolution_clock::now();
    process_standard(data);
    auto e1 = high_resolution_clock::now();
    duration<double> t1 = e1 - s1;
    cout << left << setw(20) << "Standard Access:" << t1.count() << "s" << endl;

    // Test Tiled
    auto s2 = high_resolution_clock::now();
    process_with_tiling(data);
    auto e2 = high_resolution_clock::now();
    duration<double> t2 = e2 - s2;
    cout << setw(20) << "Tiled (Blocked):" << t2.count() << "s" << endl;

    cout << "\nImprovement: " << (t1.count() / t2.count()) << "x" << endl;

    return 0;
}