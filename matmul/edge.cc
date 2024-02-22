#include <cmath>
#include <cstdbool>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <hip/hip_runtime.h>

using namespace std;
#define BLOCK_SIZE 64
#define TILE_SIZE 1
#define CHECK_HIP(cmd)                                                                                           \
    do                                                                                                           \
    {                                                                                                            \
        hipError_t error = cmd;                                                                                  \
        if (error != hipSuccess)                                                                                 \
        {                                                                                                        \
            fprintf(stderr, "HIP Error: %s (%d): %s:%d\n", hipGetErrorString(error), error, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                                                  \
        }                                                                                                        \
    } while (0)
double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}
__global__ void matmul_kernel(float *xout, float *x, float *w, int n, int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidx = threadIdx.x;
    __shared__ float W0[BLOCK_SIZE];
    __shared__ float W1[BLOCK_SIZE];
    __shared__ float W2[BLOCK_SIZE];
    __shared__ float W3[BLOCK_SIZE];
    __shared__ float W4[BLOCK_SIZE];
    __shared__ float W5[BLOCK_SIZE];
    __shared__ float X[BLOCK_SIZE];
    float sum[6] = {0};
    float zero_float = 0.;
    // float4 sum2 = make_float4(0, 0, 0, 0);
    int boundn = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int aid = 6 * idx * n;
    for (int k = 0; k < boundn; k++)
    {
        X[tidx] = k * BLOCK_SIZE + tidx < n ? x[k * BLOCK_SIZE + tidx] : 0;
        W0[tidx] = (idx * 6 < d) && k * BLOCK_SIZE + tidx < n ? w[aid] : 0;
        W1[tidx] = (idx * 6 + 1 < d) && k * BLOCK_SIZE + tidx < n ? w[aid + n] : 0;
        W2[tidx] = (idx * 6 + 2 < d) && k * BLOCK_SIZE + tidx < n ? w[aid + 2 * n] : 0;
        W3[tidx] = (idx * 6 + 3 < d) && k * BLOCK_SIZE + tidx < n ? w[aid + 3 * n] : 0;
        W4[tidx] = (idx * 6 + 4 < d) && k * BLOCK_SIZE + tidx < n ? w[aid + 4 * n] : 0;
        W5[tidx] = (idx * 6 + 5 < d) && k * BLOCK_SIZE + tidx < n ? w[aid + 5 * n] : 0;
        __syncthreads();
        for (int ex = 0; ex < BLOCK_SIZE; ex++)
        {
            float x_ = X[ex];
            float w0 = W0[ex];
            float w1 = W1[ex];
            float w2 = W2[ex];
            float w3 = W3[ex];
            float w4 = W4[ex];
            float w5 = W5[ex];
            sum[0] += x_ * w0;
            sum[1] += x_ * w1;
            sum[2] += x_ * w2;
            sum[3] += x_ * w3;
            sum[4] += x_ * w4;
            sum[5] += x_ * w5;
        }
        __syncthreads();
        aid += BLOCK_SIZE;
    }
    int left = min(6, d - idx * 6);
    for (int i = 0; i < left; i++)
    {
        xout[6 * idx + i] = sum[i];
    }
}

__global__ void matmul_kernel_shared(float *xout, float *x, float *w, int n, int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int tidx = threadIdx.x;
    __shared__ float X[BLOCK_SIZE];
    float sum = 0;
    float zero_float = 0.;
    // float4 sum2 = make_float4(0, 0, 0, 0);
    int boundn = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int aid = idx * n + tidx;
    for (int k = 0; k < boundn; k++)
    {
        X[tidx] = k * BLOCK_SIZE + tidx < n ? x[k * BLOCK_SIZE + tidx] : 0;
        __syncthreads();
        for (int ex = 0; ex < BLOCK_SIZE; ex++)
        {
            if (k * BLOCK_SIZE + ex < n && idx < d)
            {
                float x_ = X[ex];
                sum += x_ * w[idx * n + k * BLOCK_SIZE + ex];
            }
        }
        __syncthreads();
        aid += BLOCK_SIZE;
    }
    xout[idx] = sum;
}
__global__ void matmul_kernel_native(float *xout, float *x, float *w, int n, int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for (int tile = 0; tile < TILE_SIZE; tile++)
    {
        float sum = 0;
        if (idx*TILE_SIZE+tile <d)
        {
            for (int i = 0; i < n; i++)
            {
                sum += w[(idx*TILE_SIZE+tile) * n + i] * x[i];
            }
            xout[idx*TILE_SIZE+tile] = sum;
        }
    }

    
}
void matmul(float *x_cpu, float *xout, float *x, float *w, int n, int d)
{
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    float val;
    // #pragma omp parallel for num_threads(32) private(val)
    for (i = 0; i < d; i++)
    {
        val = 0.0f;
        for (int j = 0; j < n; j++)
        {
            val += w[i * n + j] * x[j];
        }
        x_cpu[i] = val;
    }

    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((d + blockDim.x - 1) / blockDim.x);
    matmul_kernel_native<<<gridDim, blockDim>>>(xout, x, w, n, d);
    CHECK_HIP(hipDeviceSynchronize());
}
void rand_mat(float *m, int R)
{
    for (int i = 0; i < R; i++)
    {
        m[i] = (float)rand() / RAND_MAX - 0.5;
    }
}
int main(int argc, char **argv)
{
    int n = 4096, d = 11008;
    float *xout, *x, *w, *x_cpu;
    CHECK_HIP(hipHostMalloc((void **)&xout, d * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipHostMalloc((void **)&x_cpu, d * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipHostMalloc((void **)&x, n * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipHostMalloc((void **)&w, d * n * sizeof(float), hipMemAllocationTypePinned));

    // rand_mat(xout,d);
    rand_mat(x, n);
    rand_mat(w, d * n);
    matmul(x_cpu, xout, x, w, n, d);
    int err_count = 0;
    for (int i = 0; i < d; i++)
    {
        if (x_cpu[i] - xout[i] > 1e-9)
        {
            printf("cpu x[%d] %8f but get gpu %8f\n", i, x_cpu[i], xout[i]);
        }
    }
    double start = get_time();
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim(((d+TILE_SIZE-1)/TILE_SIZE + blockDim.x - 1) / blockDim.x);
    matmul_kernel_native<<<gridDim, blockDim>>>(xout, x, w, n, d);
    CHECK_HIP(hipDeviceSynchronize());
    double end = get_time();
    printf("Gflop: %lf\n", 2.0 * n * d / (end - start) / 1e9);
     start = get_time();
    dim3 blockDim_(BLOCK_SIZE);
    dim3 gridDim_(((d+TILE_SIZE-1)/TILE_SIZE + blockDim.x - 1) / blockDim.x);
    matmul_kernel_shared<<<gridDim_, blockDim_>>>(xout, x, w, n, d);
    CHECK_HIP(hipDeviceSynchronize());
     end = get_time();
    printf("Gflop: %lf\n", 2.0 * n * d / (end - start) / 1e9);
    return 0;
}
