// example of measuring load imbalance with different OpenMP scheduling strategies
#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <thread>
#include <iomanip>

using namespace std;
using namespace std::chrono;

void simulate_work(int iterations, string schedule_type) {
    int n_threads = omp_get_max_threads();
    vector<double> thread_times(n_threads, 0.0);
    
    auto start_total = high_resolution_clock::now();

    // The workload is imbalanced: i=0 is fast, i=999 is slow.
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto start_thread = high_resolution_clock::now();

        // Change the schedule clause here to test different policies
        // Try: schedule(static), schedule(dynamic, 10), or schedule(guided)
        #pragma omp for schedule(runtime) 
        for (int i = 0; i < 1000; i++) {
            // Simulate variable work: higher 'i' takes longer
            // We use a small dummy math loop to keep the CPU busy
            for (int j = 0; j < i * 100; j++) {
                volatile double d = 0.1;
                d = d * d;
            }
        }

        auto end_thread = high_resolution_clock::now();
        thread_times[tid] = duration<double>(end_thread - start_thread).count();
    }

    auto end_total = high_resolution_clock::now();
    
    cout << "\nResults for: " << schedule_type << endl;
    cout << "--------------------------------------" << endl;
    for (int i = 0; i < n_threads; i++) {
        cout << "Thread " << i << " active time: " << fixed << setprecision(4) << thread_times[i] << "s" << endl;
    }
    cout << "Total Wall Clock Time: " << duration<double>(end_total - start_total).count() << "s" << endl;
}

int main() {
    cout << "Testing Load Imbalance with different OpenMP Schedules" << endl;
    
    // To test different schedules without recompiling, 
    // we use schedule(runtime) and set the environment variable.
    // Use: export OMP_SCHEDULE="static" OR "dynamic,10" OR "guided"
    
    char* env_sched = getenv("OMP_SCHEDULE");
    string current_sched = (env_sched) ? env_sched : "default (static)";
    
    simulate_work(1000, current_sched);

    return 0;
}