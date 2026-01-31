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

### Observations (Basic Programs)

- eg1: nondeterministic thread order.
- eg3: race condition; use reduction/critical.
- eg4: reduction OK (5050).
- eg5: timing decreases with threads; noisy for short runs.
- eg6-eg8: sync/scope behavior as expected.

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

Perf stat (sudo /usr/lib/linux-tools/6.8.0-90-generic/perf stat ./q1):

event syntax error: 'cpu_core/TOPDOWN.SLOTS,metric-id=cpu_core!3TOPDOWN.SLOTS!3/,cpu_core/topdown-retiring,me..'
                     \___ Bad event or PMU

Unable to find PMU or event on a PMU of 'cpu_core'

Initial error:
event syntax error: 'cpu_core/TOPDOWN.SLOTS,metric-id=cpu_core!3TOPDOWN.SLOTS!3/,cpu_core/topdown-retiring,me..'
                     \___ Cannot find PMU `cpu_core'. Missing kernel support?
Time taken: 0.024878 seconds

 Performance counter stats for './q1':

            227.92 msec task-clock                       #    6.508 CPUs utilized             
                 3      context-switches                 #   13.162 /sec                      
                 0      cpu-migrations                   #    0.000 /sec                      
               352      page-faults                      #    1.544 K/sec                     
   <not supported>      cycles                                                                
   <not supported>      instructions                                                           
   <not supported>      branches                                                               
   <not supported>      branch-misses                                                          

       0.035023413 seconds time elapsed

       0.185164000 seconds user
       0.026452000 seconds sys

### Observation

- Best with fewer threads.
- task-clock 227.92 ms.
- page-faults 352; context-switches 3.

## Q2. Matrix Multiplication (1D vs 2D Threading) 

**Matrix Size:** 500 × 500

### Output

OMP_NUM_THREADS=2
1D Parallel Time: 0.153337 seconds
2D Parallel Time: 0.124147 seconds

OMP_NUM_THREADS=4
1D Parallel Time: 0.085019 seconds
2D Parallel Time: 0.098323 seconds

Additional run :
1D Parallel Time: 0.100098
2D Parallel Time: 0.083267

Perf stat (sudo /usr/lib/linux-tools/6.8.0-90-generic/perf stat ./q2):

event syntax error: 'cpu_core/TOPDOWN.SLOTS,metric-id=cpu_core!3TOPDOWN.SLOTS!3/,cpu_core/topdown-retiring,me..'
                     \___ Bad event or PMU

Unable to find PMU or event on a PMU of 'cpu_core'

Initial error:
event syntax error: 'cpu_core/TOPDOWN.SLOTS,metric-id=cpu_core!3TOPDOWN.SLOTS!3/,cpu_core/topdown-retiring,me..'
                     \___ Cannot find PMU `cpu_core'. Missing kernel support?

 Performance counter stats for './q2':

           2010.83 msec task-clock                       #    9.933 CPUs utilized             
                38      context-switches                 #   18.898 /sec                      
                 1      cpu-migrations                   #    0.497 /sec                      
              1564      page-faults                      #  777.788 /sec                      
   <not supported>      cycles                                                                
   <not supported>      instructions                                                           
   <not supported>      branches                                                               
   <not supported>      branch-misses                                                          

       0.202446247 seconds time elapsed

       1.876625000 seconds user
       0.141335000 seconds sys


### Observation

- 2D better for low threads.
- task-clock 2010.83 ms.
- page-faults 1564; context-switches 38.

## Q3. Calculation of π Using Numerical Integration

### Output

OMP_NUM_THREADS=2
Pi = 3.1415926536
Time = 0.003116 seconds

OMP_NUM_THREADS=4
Pi = 3.1415926536
Time = 0.002822 seconds

Additional run :
Pi = 3.1415926536
Time = 0.034146 seconds

Perf stat (sudo /usr/lib/linux-tools/6.8.0-90-generic/perf stat ./q3):

event syntax error: 'cpu_core/TOPDOWN.SLOTS,metric-id=cpu_core!3TOPDOWN.SLOTS!3/,cpu_core/topdown-retiring,me..'
                     \___ Bad event or PMU

Unable to find PMU or event on a PMU of 'cpu_core'

Initial error:
event syntax error: 'cpu_core/TOPDOWN.SLOTS,metric-id=cpu_core!3TOPDOWN.SLOTS!3/,cpu_core/topdown-retiring,me..'
                     \___ Cannot find PMU `cpu_core'. Missing kernel support?
Pi = 3.1415926536
Time = 0.034146 seconds

 Performance counter stats for './q3':

            442.20 msec task-clock                       #    8.763 CPUs utilized             
                 8      context-switches                 #   18.091 /sec                      
                 2      cpu-migrations                   #    4.523 /sec                      
                95      page-faults                      #  214.836 /sec                      
   <not supported>      cycles                                                                
   <not supported>      instructions                                                           
   <not supported>      branches                                                               
   <not supported>      branch-misses                                                          

       0.050464619 seconds time elapsed

       0.337592000 seconds user
       0.105812000 seconds sys


### Observation

- π correct; scales with threads.
- task-clock 442.20 ms.
- page-faults 95; context-switches 8.

### Perf interpretation

- **Purpose:** perf measures hardware and software events (task-clock, cycles, instructions, branches, context-switches, page-faults, user/sys times).
- **PMU error:** 'cpu_core' not supported; use standard events (e.g., `cycles,instructions,cache-misses`) or enable kernel/PMU support.
- **Task-clock:** approximates CPU-time across threads (higher -> more CPU usage).
- **Context-switches / cpu-migrations:** indicate scheduler overhead; high values affect timing stability.
- **Page-faults:** indicate memory-related overhead; many page faults suggest memory pressure.
- **User / Sys times:** show time in user/kernel; compare with wall time to find blocking or I/O impact.
- **Recommendations:** run multiple trials, increase problem size for stable counters, pin threads (`taskset`) to reduce scheduling noise, and use supported perf events.

**End of LAB1 – Experimental Section**
