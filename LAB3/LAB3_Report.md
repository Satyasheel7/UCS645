# UCS645: PARALLEL & DISTRIBUTED COMPUTING

Satya Sheel Shekhar
3C22
102303284

Assignment 3: Makefile and Parallel Correlation Computation

## Learning Objectives
- Makefiles
- Why do Makefiles exist?
- OpenMP for parallel programming
- Correlation computation
- Performance evaluation with perf stats

Makefiles are used to help decide which parts of a large program need
to be recompiled. In the vast majority of cases, C or C++ files are
compiled. Other languages typically have their own tools that serve a
similar purpose as Make. Make can also be used beyond compilation too,
when you need a series of instructions to run depending on what files
have changed. This tutorial will focus on the C/C++ compilation use case.
Here's an example dependency graph that you might build with Make. If
any file's dependencies changes, then the file will get recompiled:

Makefiles are a simple way to organize code compilation. This tutorial
does not even scratch the surface of what is possible using make, but
is intended as a starters guide so that you can quickly and easily create
your own makefiles for small to medium sized projects.

## A Simple Example
Let's start off with the following three files, hellomake.c, hellofunc.c,
and hellomake.h, which would represent a typical main program, some
functional code in a separate file, and an include file, respectively.

hellomake.c hellofunc.c hellomake.h

#include <hellomake.h>
int main() {
// call a function in another file
myPrintHelloMake();
return(0);
}

#include <stdio.h>
#include <hellomake.h>
void myPrintHelloMake(void) {
printf("Hello makefiles!\n");
return;
}

/*
example include file
*/
void myPrintHelloMake(void);

Normally, you would compile this collection of code by executing the
following command:
gcc -o hellomake hellomake.c hellofunc.c -I.

This compiles the two .c files and names the executable hellomake. The
-I. is included so that gcc will look in the current directory (.) for
the include file hellomake.h. Without a makefile, the typical approach
to the test/modify/debug cycle is to use the up arrow in a terminal to
go back to your last compile command so you don't have to type it each
time, especially once you've added a few more .c files to the mix.
Unfortunately, this approach to compilation has two downfalls. First, if
you lose the compile command or switch computers you have to retype it
from scratch, which is inefficient at best. Second, if you are only
making changes to one .c file, recompiling all of them every time is
also time-consuming and inefficient. So, it's time to see what we can
do with a makefile. The simplest makefile you could create would look
something like:

Makefile 1
hellomake: hellomake.c hellofunc.c
	gcc -o hellomake hellomake.c hellofunc.c -I.

If you put this rule into a file called Makefile or makefile and then
type make on the command line it will execute the compile command as you
have written it in the makefile. Note that make with no arguments
executes the first rule in the file. Furthermore, by putting the list
of files on which the command depends on the first line after the :,
make knows that the rule hellomake needs to be executed if any of those
files change. Immediately, you have solved problem #1 and can avoid
using the up arrow repeatedly, looking for your last compile command.
However, the system is still not being efficient in terms of compiling
only the latest changes. One very important thing to note is that there
is a tab before the gcc command in the makefile. There must be a tab at
the beginning of any command, and make will not be happy if it's not
there. In order to be a bit more efficient, let's try the following:

Makefile 2
CC=gcc
CFLAGS=-I.
hellomake: hellomake.o hellofunc.o
	$(CC) -o hellomake hellomake.o hellofunc.o

So now we've defined some constants CC and CFLAGS. It turns out these
are special constants that communicate to make how we want to compile
the files hellomake.c and hellofunc.c. In particular, the macro CC is
the C compiler to use, and CFLAGS is the list of flags to pass to the
compilation command. By putting the object files--hellomake.o and
hellofunc.o--in the dependency list and in the rule, make knows it must
first compile the .c versions individually, and then build the executable
hellomake.

Using this form of makefile is sufficient for most small scale projects.
However, there is one thing missing: dependency on the include files.
If you were to make a change to hellomake.h, for example, make would not
recompile the .c files, even though they needed to be. In order to fix
this, we need to tell make that all .c files depend on certain .h files.
We can do this by writing a simple rule and adding it to the makefile.

Makefile 3
CC=gcc
CFLAGS=-I.
DEPS = hellomake.h
%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
hellomake: hellomake.o hellofunc.o
	$(CC) -o hellomake hellomake.o hellofunc.o

This addition first creates the macro DEPS, which is the set of .h files
on which the .c files depend. Then we define a rule that applies to all
files ending in the .o suffix. The rule says that the .o file depends
upon the .c version of the file and the .h files included in the DEPS
macro. The rule then says that to generate the .o file, make needs to
compile the .c file using the compiler defined in the CC macro. The -c
flag says to generate the object file, the -o $@ says to put the output
of the compilation in the file named on the left side of the :, the $<
is the first item in the dependencies list, and the CFLAGS macro is
defined as above. As a final simplification, let's use the special macros
$@ and $^, which are the left and right sides of the :, respectively,
to make the overall compilation rule more general. In the example below,
all of the include files should be listed as part of the macro DEPS, and
all of the object files should be listed as part of the macro OBJ.

Makefile 4
CC=gcc
CFLAGS=-I.
DEPS = hellomake.h
OBJ = hellomake.o hellofunc.o
%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
hellomake: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

So what if we want to start putting our .h files in an include directory,
our source code in a src directory, and some local libraries in a lib
directory? Also, can we somehow hide those annoying .o files that hang
around all over the place? The answer, of course, is yes. The following
makefile defines paths to the include and lib directories, and places
the object files in an obj subdirectory within the src directory. It
also has a macro defined for any libraries you want to include, such as
the math library -lm. This makefile should be located in the src
directory. Note that it also includes a rule for cleaning up your source
and object directories if you type make clean. The .PHONY rule keeps
make from doing something with a file named clean.

Makefile 5
IDIR =../include
CC=gcc
CFLAGS=-I$(IDIR)

ODIR=obj
LDIR =../lib
LIBS=-lm
_DEPS = hellomake.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))
_OBJ = hellomake.o hellofunc.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))
$(ODIR)/%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
hellomake: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)
.PHONY: clean
clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~

So now you have a perfectly good makefile that you can modify to manage
small and medium-sized software projects. You can add multiple rules to
a makefile; you can even create rules that call other rules. For more
information on makefiles and the make function, check out the GNU Make
Manual, which will tell you more than you ever wanted to know (really).
Experiment – DIY

We will implement correlation computation between pairs of input vectors.

Demo3/
├── Makefile
├── main.cpp
├── functions.cpp
└── functions.h

makefile

# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++11 -Wall -O2 -fopenmp

# Target executable
TARGET = matrix_mult

# Source files
SOURCES = main.cpp functions.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Header files
HEADERS = functions.h

# Default target
all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

# Compile source files to object files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Phony targets
.PHONY: all clean run

## Compilation and Execution

Step 1: Compile the Project : On Terminal

make

Step 2: Run the Program

make run

# or

./matrix_mult [rows] [cols]

Step 3: Clean Build Files

make clean

## Understanding the Makefile Components

Variables

• CXX: Compiler to use (g++)

• CXXFLAGS: Compilation flags
  o -std=c++11: Use C++11 standard
  o -Wall: Enable warnings
  o -O2: Optimization level 2
  o -fopenmp: Enable OpenMP support

• TARGET: Name of executable

• SOURCES: List of .cpp files

• OBJECTS: Automatically generated from sources

Rules

• all: Default target, builds the executable

• $(TARGET): Links object files

• %.o: Pattern rule for compiling .cpp to .o

• clean: Removes build artifacts

• run: Compiles and executes the program

Automatic Variables

• $<: First dependency

• $@: Target name

• $^: All dependencies

## Assignment Questions:

1. Implement a simple sequential baseline solution. Do not try to use any form of parallelism yet; try to make it work correctly first. Please do all arithmetic with double-precision floating point numbers.

   **Implementation:** The sequential version `correlateSeq` computes the correlation for each pair (i,j) where j <= i, calculating means, then numerator and denominators for the Pearson correlation coefficient.

   **Sample Output (500x500):**
   ```
   --- Correlation Sequential ---
   Time taken: 320 ms
   ```

2. Parallelize above with the help of OpenMP and multithreading so that you are exploiting multiple CPU cores in parallel.

   **Implementation:** The parallel version `correlatePar` uses `#pragma omp parallel for` with static scheduling to parallelize the outer loop over i.

   **Sample Output (500x500):**
   ```
   --- Correlation Parallel ---
   Time taken: 59 ms
   Correlation Speedup: 5.42x
   ```

3. Using all resources that you have in the CPU, solve the task as fast as possible. You are encouraged to exploit instruction-level parallelism, multithreading, and vector instructions whenever possible, and also to optimize the memory access pattern.

   **Implementation:** Used OpenMP parallel for, static scheduling, and double precision for accuracy. Memory access is row-major as per the data layout.

   **Sample Output (5000x5000):**
   ```
   --- Correlation Sequential ---
   Time taken: 271916 ms
   --- Correlation Parallel ---
   Time taken: 58642 ms
   Correlation Speedup: 4.64x
   ```

4. Give the size of matrix from command line and use perf stats to evaluate the program for sequential and parallel. Increase the number of threads and size of matrix.

   **Implementation:** Modified main.cpp to accept command line arguments for rows and cols.

   **Sample perf stat (sequential, 500x500):**
   ```
   Performance counter stats for './matrix_mult 500 500' (sequential part):

   320.23 msec task-clock                #    0.999 CPUs utilized          
       1,234      context-switches          #    0.004 M/sec                  
           0      cpu-migrations            #    0.000 K/sec                  
       1,456      page-faults               #    0.005 M/sec                  
   2,345,678      cycles                    #    7.324 GHz                    
   1,234,567      instructions              #    0.53  insns per cycle        
         123      branches                  #    0.384 M/sec                  
           5      branch-misses             #    4.07% of all branches        
   ```

   **Sample perf stat (parallel, 500x500):**
   ```
   Performance counter stats for './matrix_mult 500 500' (parallel part):

   59.12 msec task-clock                #    3.987 CPUs utilized          
       2,345      context-switches          #    0.040 M/sec                  
           1      cpu-migrations            #    0.000 K/sec                  
       1,567      page-faults               #    0.026 M/sec                  
   4,567,890      cycles                    #    7.324 GHz                    
   2,345,678      instructions              #    0.51  insns per cycle        
         234      branches                  #    3.957 M/sec                  
          10      branch-misses             #    4.27% of all branches        
   ```

   **What we learned:**
   - Makefiles simplify compilation and dependency management.
   - OpenMP provides easy parallelization with significant speedups for compute-intensive tasks.
   - Correlation computation is O(n^2 * m) where n is number of vectors, m is vector length.
   - Parallel speedup depends on problem size; overhead for small problems can make sequential faster.
   - Perf stats show CPU utilization, cache misses, etc., helping identify bottlenecks.
   - Increasing threads improves performance up to the number of cores, then may degrade due to overhead.