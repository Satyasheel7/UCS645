// example of false sharing and its mitigation using padding in C++ with OpenMP
//  This code demonstrates the performance impact of false sharing
// and how padding can be used to mitigate it.
// False sharing occurs when threads on different processors modify variables
// that reside on the same cache line, leading to performance degradation.
// This example compares performance with and without padding.
#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

// 64 bytes is the standard cache line size for most modern CPUs
#define CACHE_LINE_SIZE 64 

struct BadStruct {
    long long val; // Only 8 bytes
};

struct PaddedStruct {
    long long val; // 8 bytes
    // We add 56 bytes of "junk" to ensure no two 'val' fields sit on the same cache line
    char padding[CACHE_LINE_SIZE - sizeof(long long)]; 
};

void run_test(bool use_padding, long long iterations) {
    int n_threads = omp_get_max_threads();
    vector<BadStruct> bad_data(n_threads, {0});
    vector<PaddedStruct> good_data(n_threads, {0});

    auto start = high_resolution_clock::now();

    #pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        for (long long i = 0; i < iterations; i++) {
            if (use_padding)
                good_data[tid].val += 1;
            else
                bad_data[tid].val += 1;
        }
    }

    auto end = high_resolution_clock::now();
    duration<double> diff = end - start;
    
    cout << (use_padding ? "Padded (Correct Mitigation): " : "False Sharing (The Problem): ") 
              << diff.count() << "s" << endl;
}

int main() {
    long long iterations = 100000000; // 100 Million
    cout << "Testing False Sharing on " << omp_get_max_threads() << " threads" << endl;
    cout << "----------------------------------------------" << endl;
    
    run_test(false, iterations); // High cache contention
    run_test(true, iterations);  // Independent cache lines
    
    return 0;
}