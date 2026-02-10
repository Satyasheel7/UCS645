// example of strong and weak scaling in C++ using OpenMP
// measuring performance of Pi calculation using numerical integration
// as we vary the number of threads and problem size

#include <iostream>
#include <omp.h>
#include <chrono>
#include <vector>
#include <iomanip>
using namespace std;
// Performance measurement function
double calculate_pi_parallel(long long steps, int num_threads) {
    double step = 1.0 / (double)steps;
    double sum = 0.0;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(num_threads)
    {
        double x, local_sum = 0.0;

        #pragma omp for
        for (long long i = 0; i < steps; i++) {
            x = (i + 0.5) * step;
            local_sum += 4.0 / (1.0 + x * x);
        }

        #pragma omp atomic
        sum += local_sum;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

int main() {
    int max_cores = 23;
    long long base_steps = 100000000; // 100 Million steps
    cout << fixed << setprecision(4);
    // --- SECTION 1: STRONG SCALING ---
    // Problem size is FIXED at 500M steps
    long long strong_total_steps = 500000000;
    cout << "--- STRONG SCALING (Fixed Total Work: " << strong_total_steps << ") ---" << endl;
    cout << left << setw(12) << "Cores" << setw(15) << "Time (s)" << "Speedup" << endl;    
    double t_serial_strong = calculate_pi_parallel(strong_total_steps, 1);
    cout << setw(12) << "1 (Serial)" << setw(15) << t_serial_strong << "1.00x" << endl;
    for (int n = 4; n <= max_cores; n += 4) {
        double p_time = calculate_pi_parallel(strong_total_steps, n);
        cout << setw(4) << n << " Cores" << setw(5) << "" 
             << setw(15) << p_time << (t_serial_strong / p_time) << "x" << endl;
    }
    cout << "\n";
    // --- SECTION 2: WEAK SCALING ---
    // Problem size grows with core count (100M steps PER CORE)
    cout << "--- WEAK SCALING (Fixed Work Per Core: " << base_steps << ") ---" << endl;
    cout << left << setw(12) << "Cores" << setw(15) << "Total Work" << setw(15) << "Time (s)" << "Efficiency" << endl;
    double t_serial_weak = calculate_pi_parallel(base_steps, 1);
    cout << setw(12) << "1 (Serial)" << setw(15) << base_steps << setw(15) << t_serial_weak << "100%" << endl;
    for (int n = 4; n <= max_cores; n += 4) {
        long long current_work = base_steps * n;
        double p_time = calculate_pi_parallel(current_work, n);        
        // Efficiency = (T1 / Tn) * 100. In ideal weak scaling, Tn remains equal to T1.
        double efficiency = (t_serial_weak / p_time) * 100.0;        
        cout << setw(4) << n << " Cores" << setw(5) << "" 
             << setw(15) << current_work << setw(15) << p_time << efficiency << "%" << endl;
    }
    return 0;
}