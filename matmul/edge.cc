#include <cmath>
#include <cstdbool>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <hip/hip_runtime.h>

using namespace std;
#define BLOCK_SIZE 16
#define TILE_SIZE 1
#define BATCH 2
#define WARP_SIZE 64
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

// __global__ void matmul_kernel_shared(float *xout, float *x, float *w, int n, int d)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     int tidx = threadIdx.x;
//     __shared__ float X[BLOCK_SIZE];
//     float sum = 0;
//     float zero_float = 0.;
//     // float4 sum2 = make_float4(0, 0, 0, 0);
//     int boundn = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     int aid = idx * n + tidx;
//     for (int k = 0; k < boundn; k++)
//     {
//         X[tidx] = k * BLOCK_SIZE + tidx < n ? x[k * BLOCK_SIZE + tidx] : 0;
//         __syncthreads();
//         for (int ex = 0; ex < BLOCK_SIZE; ex++)
//         {
//             if (k * BLOCK_SIZE + ex < n && idx < d)
//             {
//                 float x_ = X[ex];
//                 sum += x_ * w[idx * n + k * BLOCK_SIZE + ex];
//             }
//         }
//         __syncthreads();
//         aid += BLOCK_SIZE;
//     }
//     xout[idx] = sum;
// }
__device__ float warpReduceSum(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor(val, offset);
    return val;
}

__device__ float blockReduceSum(float val)
{
    static __shared__ float shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    val = warpReduceSum(val);
    if (lane == 0)
        shared[wid] = val;

    __syncthreads();
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

    if (wid == 0)
        val = warpReduceSum(val);
}
__global__ void matmul_kernel_native(float *xout, float *x, float *w, int n, int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;

#pragma unroll 23
    for (int i = 0; i < n; i++)
    {
        sum += w[idx * n + i] * x[i];
    }
    xout[idx] = sum;
}

__global__ void matmul_kernel_native_batch(float *xout, float *x, float *w, int n, int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float sum[BATCH] = {0};

#pragma unroll 23
    for (int i = 0; i < n; i++)
    {
// #pragma unroll BATCH
        for (int b = 0; b < BATCH; b++)
        {
            sum[b] += w[idx * n + i] * x[b * n + i];
        }
    }
// #pragma unroll BATCH
    for (int b = 0; b < BATCH; b++)
        xout[b * d + idx] = sum[b];
}
__global__ void matmul_kernel_shared_f4(float *xout, float4 *x, float4 *w, int n, int d)
{
    __shared__ float4 x_shared[BLOCK_SIZE + 1];
    // __shared__ float4 w_shared[BLOCK_SIZE + 1];
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    float sum1 = 0.0f;
    int j = 0;
    for (; j < n; j += BLOCK_SIZE)
    {
        x_shared[threadIdx.x] = x[j + threadIdx.x];

        __syncthreads();

        for (int k = 0; i < d && k < BLOCK_SIZE; k++)
        {
            // float4 x_shared =
            sum += w[i * n + j + k].x * x_shared[k].x;
            sum += w[i * n + j + k].y * x_shared[k].y;
            sum += w[i * n + j + k].z * x_shared[k].z;
            sum += w[i * n + j + k].w * x_shared[k].w;
        }
        __syncthreads();
    }

    if (i >= d)
        return;
    // if (threadIdx.y == 0)
    xout[i] = sum;
}
__global__ void matmul_kernel_shared(float *xout, float *x, float *w, int n, int d)
{
    __shared__ float x_shared[BLOCK_SIZE + 1];
    // __shared__ float w_shared[BLOCK_SIZE + 1];
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    float sum1 = 0.0f;
    int j = 0;
    for (; j < n; j += BLOCK_SIZE)
    {
        x_shared[threadIdx.x] = x[j + threadIdx.x];

        __syncthreads();

        for (int k = 0; i < d && k < BLOCK_SIZE; k++)
        {
            // float4 x_shared =
            sum += w[i * n + j + k] * x_shared[k];
        }

        __syncthreads();
    }

    if (i >= d)
        return;
    if (threadIdx.y == 0)
        xout[i] = sum;
}

__global__ void matmul_kernel_shared_batch(float *xout, float4 *x, float4 *w, int n, int d)
{
    __shared__ float4 x_shared[BATCH][BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float sum[BATCH] = {0.0f};
    int j = 0;
    for (; j < n; j += BLOCK_SIZE)
    {

        for (int b = 0; b < BATCH; b++)
            x_shared[b][threadIdx.x] = x[b * n + j + threadIdx.x];
        __syncthreads();
        for (int k = 0; i < d && k < BLOCK_SIZE; k++)
        {
            // float4 w4 = w[i * n + j + k];

            for (int b = 0; b < BATCH; b++)
            {
                sum[b] += w[i * n + j + k].x * x_shared[b][k].x;
                sum[b] += w[i * n + j + k].y * x_shared[b][k].y;
                sum[b] += w[i * n + j + k].z * x_shared[b][k].z;
                sum[b] += w[i * n + j + k].w * x_shared[b][k].w;
            }
        }
        __syncthreads();
    }

    //     for (; j < n; j++)
    //     {
    //         float4 w4 = w[i * n + j];
    // #pragma unroll BATCH
    //         for (int b = 0; b < BATCH; b++)
    //         {
    //             sum[b] += w4.x * x[b * n + j].x;
    //             sum[b] += w4.y * x[b * n + j].y;
    //             sum[b] += w4.z * x[b * n + j].z;
    //             sum[b] += w4.w * x[b * n + j].w;
    //         }
    //     }
    if (i >= d)
        return;
    // #pragma unroll BATCH
    for (int b = 0; b < BATCH; b++)
        xout[b * d + i] = sum[b];
}
void matmul(float *x_cpu, float *xout, float *x, float *w, int n, int d)
{
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    float val;
    // #pragma omp parallel for num_threads(32) private(val)
    for (int b = 0; b < BATCH; b++)
        for (i = 0; i < d; i++)
        {
            val = 0.0f;
            for (int j = 0; j < n; j++)
            {
                val += w[i * n + j] * x[b * n + j];
            }
            x_cpu[b * d + i] = val;
        }

    dim3 blockDim(BLOCK_SIZE, TILE_SIZE);
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
    int n = 11008, d = 4096;
    int batch = 4;
    float *xout, *x, *w, *x_cpu, *x_gpu, *w_gpu, *x_out_gpu;
    CHECK_HIP(hipHostMalloc((void **)&xout, d * BATCH * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipHostMalloc((void **)&x_cpu, d * BATCH * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipHostMalloc((void **)&x, n * BATCH * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipHostMalloc((void **)&w, d * n * sizeof(float), hipMemAllocationTypePinned));

    CHECK_HIP(hipMalloc((void **)&x_gpu, BATCH * n * sizeof(float)));
    CHECK_HIP(hipMalloc((void **)&w_gpu, d * n * sizeof(float)));
    CHECK_HIP(hipMalloc((void **)&x_out_gpu, BATCH * d * sizeof(float)));
    CHECK_HIP(hipDeviceSynchronize());

    // rand_mat(xout,d);
    rand_mat(x, n * BATCH);
    rand_mat(w, d * n);

    //     for (int i =0;i<n;i++)
    //         printf("x[%d]: %f\n",i,x[i]);
    //  for (int j =0;j<d;j++)
    //     for (int i =0;i<n;i++)

    //         printf("w[%d][%d]: %f\n",j,i,w[j*n+i]);

    CHECK_HIP(hipMemcpy(x_gpu, x, n * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(w_gpu, w, n * d * sizeof(float), hipMemcpyHostToDevice));
    matmul(x_cpu, xout, x, w, n, d);
    int err_count = 0;
    for (int i = 0; i < d * BATCH; i++)
    {
        if (x_cpu[i] - xout[i] > 1e-5)
        {
            printf("cpu x[%d] %8f but get gpu %8f\n", i, x_cpu[i], xout[i]);
        }
    }
    // double start = get_time();
    // dim3 blockDim(BLOCK_SIZE);
    // dim3 gridDim(((d + TILE_SIZE - 1) / TILE_SIZE + blockDim.x - 1) / blockDim.x);
    // matmul_kernel_native<<<gridDim, blockDim>>>(xout, x, w, n, d);
    // CHECK_HIP(hipDeviceSynchronize());
    // double end = get_time();
    // printf("Gflop: %lf\n", 2.0 * n * d / (end - start) / 1e9);
    // start = get_time();
    dim3 blockDim_(BLOCK_SIZE, TILE_SIZE);
    dim3 gridDim_(BATCH * (d + blockDim_.x - 1) / blockDim_.x);
    // matmul_kernel_shared<<<gridDim_, blockDim_>>>(xout, (float4 *)x, (float4 *)w, (n+3)/4, d);
    // CHECK_HIP(hipDeviceSynchronize());
    // end = get_time();
    // printf("Gflop: %lf\n", 2.0 * n * d / (end - start) / 1e9);
    double start_ = get_time();

    matmul_kernel_native<<<gridDim_, blockDim_>>>(x_out_gpu, x_gpu, w_gpu, n, d);

    CHECK_HIP(hipDeviceSynchronize());
    double end_ = get_time();
    printf("total time cost %lf\n", end_ - start_);
    printf("Gflop: %lf\n", 2.0 * n * d  / (end_ - start_) / 1e9);

    // start_ = get_time();

    // matmul_kernel_shared_f4<<<gridDim_, blockDim_>>>(x_out_gpu, (float4 *)x_gpu, (float4 *)w_gpu, n / 4, d);

    // CHECK_HIP(hipDeviceSynchronize());
    // end_ = get_time();
    // printf("total time cost %lf\n", end_ - start_);
    // printf("Gflop: %lf\n", 2.0 * n * d  / (end_ - start_) / 1e9);

    // double start_ = get_time();

    // matmul_kernel_native_batch<<<gridDim_, blockDim_>>>(x_out_gpu,x_gpu, w_gpu, n , d);

    // CHECK_HIP(hipDeviceSynchronize());
    // double end_ = get_time();
    // printf("total time cost %lf\n", end_ - start_);
    // printf("Gflop: %lf\n", 2.0 * n * d * BATCH / (end_ - start_) / 1e9);

    return 0;
}
