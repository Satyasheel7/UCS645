// Experiment: Static vs. Dynamic Scheduling to Handle Load Imbalance
// Description: This code demonstrates how static scheduling can lead to load imbalance
// in parallel loops with varying workloads, and how dynamic scheduling can help mitigate this issue.
// The workload increases with the loop index, simulating a scenario where
#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <thread>
#include <cmath>
#include <string>

using namespace std;
using namespace std::chrono;
// Simulates work that gets progressively harder
void heavy_work(int iterations) {
    // Artificial delay to simulate complex math
    // Lower iterations = fast, Higher iterations = slow
    double dummy = 0;
    for (int i = 0; i < iterations * 1000; ++i) {
        dummy += sin(i) * cos(i);
    }
}
void run_test(string label, bool use_dynamic) {
    int num_tasks = 1000;
    auto start = high_resolution_clock::now();
    // The "schedule" clause is the key here
    if (use_dynamic) {
        // Chunk size of 10 allows cores to grab small batches of work as they become free
        #pragma omp parallel for schedule(dynamic, 10)
        for (int i = 0; i < num_tasks; i++) {
            heavy_work(i); // Workload increases as i increases
        }
    } else {
        // Divide tasks equally (1000 / 23) regardless of how hard they are
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_tasks; i++) {
            heavy_work(i); 
        }
    }
    auto end = high_resolution_clock::now();
    duration<double> diff = end - start;
    cout << label << " Time: " << diff.count() << "s" << endl;
}
int main() {
    cout << "Testing Load Imbalance on " << omp_get_max_threads() << " cores..." << endl;
    cout << "------------------------------------------" << endl;
    // 1. Static: Cores 0-5 get the easy start, higher cores get the brutal end.
    // Result: Most cores will finish in milliseconds and wait at the barrier for the last core.
    run_test("Static Scheduling (Fixed Chunks) ", false);
    // 2. Dynamic: Faster cores help finish the heavy tasks at the end.
    // Result: Higher CPU utilization across all 23 cores.
    run_test("Dynamic Scheduling (Work Stealing)", true);
    return 0;
}