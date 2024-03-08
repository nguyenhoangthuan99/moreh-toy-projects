#include <cmath>
#include <cstdbool>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <hip/hip_runtime.h>

using namespace std;
#define BLOCK_SIZE 8
#define TILE_SIZE 1
#define BATCH 8
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

__device__ float warpReduceSum(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor(val, offset);
    return val;
}

__device__ float blockReduceSum(float val)
{
    static __shared__ float shared[WARP_SIZE + 1];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    val = warpReduceSum(val);
    if (lane == 0)
        shared[wid] = val;

    __syncthreads();
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

    if (wid == 0)
        val = warpReduceSum(val);
    return val;
}
__global__ void matmul_kernel_block(float *xout, float4 *x, float4 *w, int n, int d)
{
    int index = blockIdx.x;
    int block = blockDim.x;
    // if (threadIdx.x == 0)
    //     printf("index %d %d\n", index, block);

    if (index >= d)
        return;
    float sum = 0;
#pragma unroll 20
    for (int i = threadIdx.x; i < n; i += block)
    {
        sum += w[index * n + i].x * x[i].x;
        sum += w[index * n + i].y * x[i].y;
        sum += w[index * n + i].z * x[i].z;
        sum += w[index * n + i].w * x[i].w;
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0)
        xout[index] = sum;
}
__global__ void matmul_kernel_block_batch(float *xout, float4 *x, float4 *w, int n, int d)
{
    int index = blockIdx.x;
    int block = blockDim.x;
    // if (threadIdx.x == 0)
    //     printf("index %d %d\n", index, block);

    if (index >= d)
        return;
    float sum[BATCH] = {0};
#pragma unroll 16
    for (int i = threadIdx.x; i < n; i += block)
    {
        float4 w4 = w[index * n + i];
        for (int b = 0; b < BATCH; b++)
        {
            sum[b] += w4.x * x[b * n + i].x;
            sum[b] += w4.y * x[b * n + i].y;
            sum[b] += w4.z * x[b * n + i].z;
            sum[b] += w4.w * x[b * n + i].w;
        }
    }
    for (int b = 0; b < BATCH; b++)
        sum[b] = blockReduceSum(sum[b]);
    if (threadIdx.x == 0)
    {
        for (int b = 0; b < BATCH; b++)
            xout[b * d + index] = sum[b];
    }
}

__global__ void matmul_kernel(float *xout, float4 *x, float4 *w, int n, int d)
{
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= d)
        return;
    float sum = 0;
#pragma unroll 16
    for (int i = threadIdx.x; i < n; i += WARP_SIZE)
    {
        sum += w[index * n + i].x * x[i].x;
        sum += w[index * n + i].y * x[i].y;
        sum += w[index * n + i].z * x[i].z;
        sum += w[index * n + i].w * x[i].w;
    }
    sum = warpReduceSum(sum);
    if (threadIdx.x == 0)
        xout[index] = sum;
}
__global__ void matmul_kernel_batch(float *xout, float4 *x, float4 *w, int n, int d)
{
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= d)
        return;
    float sum[BATCH] = {0};

#pragma unroll 16
    for (int i = threadIdx.x; i < n; i += WARP_SIZE)
    {
        float4 w4 = w[index * n + i];
        for (int b = 0; b < BATCH; b++)
        {
            sum[b] += w4.x * x[b * n + i].x;
            sum[b] += w4.y * x[b * n + i].y;
            sum[b] += w4.z * x[b * n + i].z;
            sum[b] += w4.w * x[b * n + i].w;
        }
    }
    for (int b = 0; b < BATCH; b++)
        sum[b] = warpReduceSum(sum[b]);
    if (threadIdx.x == 0)
    {
        for (int b = 0; b < BATCH; b++)
            xout[b * d + index] = sum[b];
    }
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

    dim3 blockDim(WARP_SIZE, BLOCK_SIZE);
    dim3 gridDim((d + blockDim.y - 1) / blockDim.y);
    // matmul_kernel_batch<<<gridDim, blockDim>>>(xout, (float4 *)x, (float4 *)w, n/4 , d);
    matmul_kernel_block_batch<<<d, 512>>>(xout, (float4 *)x, (float4 *)w, n / 4, d);
    CHECK_HIP(hipDeviceSynchronize());
    // exit(0);
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
    int n = 5120, d = 13824;
    int batch = 4;
    float *xout, *x, *w, *x_cpu, *x_gpu, *w_gpu, *x_out_gpu;
    CHECK_HIP(hipHostMalloc((void **)&xout, d * BATCH * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipHostMalloc((void **)&x_cpu, d * BATCH * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipHostMalloc((void **)&x, n * BATCH * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipHostMalloc((void **)&w, d * n * sizeof(float), hipMemAllocationTypePinned));

    CHECK_HIP(hipMalloc((void **)&x_gpu, BATCH * n * sizeof(float)));//, hipMemAllocationTypePinned));
    CHECK_HIP(hipMalloc((void **)&w_gpu, d * n * sizeof(float)));
    CHECK_HIP(hipMalloc((void **)&x_out_gpu, BATCH * d * sizeof(float)));//, hipMemAllocationTypePinned));
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
            if (xout[i] - 0 < 1e-4)
                err_count += 1;
            printf("cpu x[%d] %8f but get gpu %8f\n", i, x_cpu[i], xout[i]);
        }
    }
    // printf("error count %d\n", err_count);
    // exit(0);
    dim3 blockDim_(WARP_SIZE, BLOCK_SIZE);
    dim3 gridDim_(BATCH * (d + blockDim_.y - 1) / blockDim_.y);

    double start_ = get_time();

    // matmul_kernel_batch<<<gridDim_, blockDim_>>>(x_out_gpu,(float4 *) x_gpu,(float4 *) w_gpu, n /4, d);
    matmul_kernel_block_batch<<<d, 512>>>(x_out_gpu, (float4 *)x_gpu, (float4 *)w_gpu, n / 4, d);
    CHECK_HIP(hipDeviceSynchronize());
    double end_ = get_time();
    printf("total time cost %lf\n", end_ - start_);
    printf("Gflop: %lf\n", 2.0 * n * d * BATCH / (end_ - start_) / 1e9);

    start_ = get_time();

    matmul_kernel_batch<<<gridDim_, blockDim_>>>(x_out_gpu, (float4 *)x_gpu, (float4 *)w_gpu, n / 4, d);
    // matmul_kernel<<<d, 512>>>(x_out_gpu, (float4 *)x_gpu, (float4 *)w_gpu, n/4, d);
    CHECK_HIP(hipDeviceSynchronize());
    end_ = get_time();
    printf("total time cost %lf\n", end_ - start_);
    printf("Gflop: %lf\n", 2.0 * n * d * BATCH / (end_ - start_) / 1e9);

    return 0;
}
