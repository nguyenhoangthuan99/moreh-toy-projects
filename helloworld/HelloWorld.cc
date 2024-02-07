#include <hip/hip_runtime.h>

#define CHECK_HIP(cmd) do { \
  hipError_t error = cmd; \
  if (error != hipSuccess) { \
    fprintf(stderr, "HIP Error: %s (%d): %s:%d\n", hipGetErrorString(error), error, __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

__global__ void hello() {
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;


    printf("Hello from thread %d\n", threadId);    
}

int main() {
    
    hello<<<3, 3>>>();
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    return 0;
}