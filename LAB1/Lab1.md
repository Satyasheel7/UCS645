# UCS645 – LAB1 (OpenMP Experiments)

Student:Satya Sheel Shekhar
Username:satyasheel07

## BASIC PROGRAMS – OUTPUTS

### eg1 – Hello World
Hello from thread 10
Hello from thread 0
Hello from thread 4
Hello from thread 9
Hello from thread 8
Hello from thread 7
Hello from thread 2
Hello from thread 11
Hello from thread 3
Hello from thread 6
Hello from thread 5
Hello from thread 1

### eg2 – Controlling Number of Threads

Thread ID: 2
Thread ID: 3
Thread ID: 1
Thread ID: 0

### eg3 – Parallel FOR Loop (Race Condition)

Sum = 2629

### eg4 – Reduction Clause

Correct Sum = 5050

### eg5 – Measuring Execution Time

Time taken: 0.156045 seconds
OMP_NUM_THREADS=4
Time taken: 0.118397 seconds

### eg6 – Scheduling in OpenMP

Thread 0 -> i = 0
Thread 0 -> i = 1
Thread 10 -> i = 4
Thread 10 -> i = 5
Thread 8 -> i = 2
Thread 8 -> i = 3
Thread 1 -> i = 8
Thread 1 -> i = 9
Thread 7 -> i = 12
Thread 7 -> i = 13
Thread 11 -> i = 14
Thread 11 -> i = 15
Thread 9 -> i = 6
Thread 9 -> i = 7
Thread 2 -> i = 10
Thread 2 -> i = 11

### eg7 – Critical Section

Thread 11 in critical section
Thread 5 in critical section
Thread 6 in critical section
Thread 8 in critical section
Thread 2 in critical section
Thread 0 in critical section
Thread 10 in critical section
Thread 7 in critical section
Thread 9 in critical section
Thread 4 in critical section
Thread 1 in critical section
Thread 3 in critical section

### eg8 – Private vs Shared Variables

Thread 2: x = 2
Thread 7: x = 7
Thread 10: x = 10
Thread 6: x = 6
Thread 5: x = 5
Thread 11: x = 11
Thread 8: x = 8
Thread 9: x = 9
Thread 1: x = 1
Thread 0: x = 0
Thread 4: x = 4
Thread 3: x = 3

## EXPERIMENTAL QUESTIONS & RESULTS

## Q1. DAXPY Loop

**Operation:**
X[i] = a × X[i] + Y[i], where X and Y are vectors of size 2^16.

### Output

OMP_NUM_THREADS=2
Time taken: 0.000534 seconds

OMP_NUM_THREADS=4
Time taken: 0.000937 seconds

OMP_NUM_THREADS=8
Time taken: 0.002539 seconds
### Observation

Best performance is achieved with a smaller number of threads.
Execution time increases when threads exceed the number of available CPU cores.

## Q2. Matrix Multiplication (1D vs 2D Threading)

**Matrix Size:** 500 × 500

### Output

OMP_NUM_THREADS=2
1D Parallel Time: 0.153337 seconds
2D Parallel Time: 0.124147 seconds

OMP_NUM_THREADS=4
1D Parallel Time: 0.085019 seconds
2D Parallel Time: 0.098323 seconds

### Observation

2D threading performs better for smaller thread counts.
With higher threads, overhead may reduce the benefit of nested parallelism.

## Q3. Calculation of π Using Numerical Integration

### Output

OMP_NUM_THREADS=2
Pi = 3.1415926536
Time = 0.003116 seconds

OMP_NUM_THREADS=4
Pi = 3.1415926536
Time = 0.002822 seconds

### Observation

Correct value of π is obtained for all thread counts.
Execution time decreases as number of threads increases.

**End of LAB1 – Experimental Section**
