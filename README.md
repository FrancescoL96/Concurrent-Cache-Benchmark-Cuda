# Concurrent Cache Benchmark Cuda
Computes an array with CPU and GPU, without race conditions. The computation will be much faster on Zero Copy when using the Jetson Xavier.
## Usage
Simply compile by running the compile script (currently configured for Volta/Xavier) on a Jetson Board:
```
./compile
```
and then run with
```
./conc_bench
```
## Configuration
To modify the configuration change the variables contained at the start of "conc_bench.cu", comments show what they do:
```
// Set PRINT to 1 for debug output
#define PRINT 0
#define FROM_debug 0
#define TO_debug 16

// Set ZEROCOPY to 1 to use Zero Copy Memory Mode, UNIFIED to 1 to use Unified Memory, COPY to 1 to use Copy
#define ZEROCOPY 1
#define UNIFIED 0
#define COPY 0

// Set RESULTCHECK to 1 to verify the result with a single CPU thread
#define RESULTCHECK 1

// Set CPU to 1 to use the CPU concurrently
#define CPU 1
// Set OPENMP to 1 to use more than 1 thread for the CPU
#define OPENMP 1

unsigned int N = 2;
const int POW = 27;			 // Maximum is 30, anything higher and the system will use swap, making the Cuda kernels crash
const int RUNS = 50;		// How many times the benchmark is run
const int SUMS = 10;		// As CPU and GPU work on either the left side or right side, this number indicates how many "side swaps" there will be
const int BLOCK_SIZE_X = 1024; // Cuda Block Size
```
