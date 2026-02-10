// Experiment: Measuring Load Imbalance with Different OpenMP Scheduling Strategies
// Description: This code demonstrates how different OpenMP scheduling strategies
// affect load imbalance in parallel loops with varying workloads.
// The workload increases with the loop index, simulating a scenario where
// some iterations take significantly longer than others.
// C++ Code: Load Imbalance Measurement with Various Schedules
#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <cmath>
using namespace std;
using namespace std::chrono;
// A workload that gets progressively harder
void work(int i) {
    double dummy = 0;
    // Workload is proportional to i to create imbalance
    long long limit = (i + 1) * 200000; 
    for (long long j = 0; j < limit; ++j) {
        dummy += sin(j) * cos(j);
    }
}

void measure_imbalance(string schedule_name, int n_threads, int N) {
    vector<double> thread_times(n_threads, 0.0);
    auto start_wall = high_resolution_clock::now();
    // We use a manual parallel block to track time per thread
    #pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        auto start_thread = high_resolution_clock::now();
        // The specific loop we are testing
        if (schedule_name == "static") {
            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) work(i);
        } else if (schedule_name == "dynamic,4") {
            #pragma omp for schedule(dynamic, 4)
            for (int i = 0; i < N; i++) work(i);
        } else if (schedule_name == "guided") {
            #pragma omp for schedule(guided)
            for (int i = 0; i < N; i++) work(i);
        }
        auto end_thread = high_resolution_clock::now();
        duration<double> diff = end_thread - start_thread;
        thread_times[tid] = diff.count();
    }
    auto end_wall = high_resolution_clock::now();
    duration<double> wall_time = end_wall - start_wall;
    // Calculate Metrics
    double t_max = *max_element(thread_times.begin(), thread_times.end());
    double t_sum = accumulate(thread_times.begin(), thread_times.end(), 0.0);
    double t_avg = t_sum / n_threads;
    double imbalance = (t_max - t_avg) / t_avg;
    // Output results
    cout << left << setw(15) << schedule_name 
         << setw(12) << t_max << "s"
         << setw(12) << t_avg << "s"
         << fixed << setprecision(2) << (imbalance * 100) << "%" << endl;
}
int main() {
    int N = 2000; // Total iterations
    int n_threads = omp_get_max_threads();
    cout << "Running on " << n_threads << " threads...\n";
    cout << left << setw(15) << "Schedule" 
         << setw(13) << "T_max" 
         << setw(13) << "T_avg" 
         << "Imbalance (%)" << endl;
    cout << string(55, '-') << endl;
    measure_imbalance("static", n_threads, N);
    measure_imbalance("dynamic,4", n_threads, N);
    measure_imbalance("guided", n_threads, N);
    return 0;
}