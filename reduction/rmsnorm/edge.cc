#include <cmath>
#include <cstdbool>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <hip/hip_runtime.h>

using namespace std;
#define BLOCK_SIZE 256
#define TILE_SIZE 1
#define BATCH 4
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
    {
        val += __shfl_down(val, offset);
    }

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

void rmsnorm_cpu(float *o, float *x, float *weight, int size)
{
    // calculate sum of squares
    for (int b = 0; b < BATCH; b++)
    {
        float ss = 0.0f;
        // #pragma omp parallel for num_threads(32) reduction(+ : ss)
        for (int j = 0; j < size; j++)
        {
            ss += x[b*size+j] * x[b*size+j];
        }
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        // normalize and scale
        for (int j = 0; j < size; j++)
        {
            o[b*size+j] = weight[j] * (ss * x[b*size+j]);
        }
    }
}

__global__ void rmsnorm_kernel(float *o, float *x, float *weight, int size)
{
    // parallel reduction of sum of squares via CUB
    // __shared__ float ss_sum[BLOCK_SIZE + 1];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < size; i += BLOCK_SIZE)
    {
        ss += x[i] * x[i];
    }

    ss = blockReduceSum(ss);
    // __shared__ float shared_ss;
    // if (threadIdx.x == 0)
    // {
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
    //     shared_ss = ss;
    // }
    // __syncthreads();
    // ss = shared_ss;

    //   ss_sum[threadIdx.x] = ss;
    //   __syncthreads();
    //   float sum_ss = 0.0f;

    //   for (int i = 0; i < BLOCK_SIZE; i++)
    //   {
    //     sum_ss += ss_sum[i];
    //   }
    //   sum_ss /=size;
    //   sum_ss += 1e-5f;
    //   sum_ss = 1.0f / __fsqrt_rn(sum_ss);
    //   ss = sum_ss;

    // normalize and scale
    for (int j = threadIdx.x; j < size; j += BLOCK_SIZE)
    {
        // int j = threadIdx.x + i * num_threads_lrg;
        // if (j < size)
        // {
        float a = weight[j] * (ss * x[j]);
        o[j] = a;
        // }
    }
}
__global__ void rmsnorm_kernel_batch(float *o, float *x, float *weight, int size)
{
    // parallel reduction of sum of squares via CUB
    //   __shared__ float ss_sum[BLOCK_SIZE+1];
    int batch_id = blockIdx.x;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < size; i += BLOCK_SIZE)
    {
        ss += x[batch_id * size + i] * x[batch_id * size + i];
    }

    ss = blockReduceSum(ss);
    __shared__ float shared_ss;
    if (threadIdx.x == 0)
    {
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        shared_ss = ss;
    }
    __syncthreads();
    ss = shared_ss;

    // normalize and scale
    for (int j = threadIdx.x; j < size; j += BLOCK_SIZE)
    {
        // int j = threadIdx.x + i * num_threads_lrg;
        // if (j < size)
        // {
        float a = weight[j] * (ss * x[batch_id * size + j]);
        o[batch_id * size + j] = a;
        // }
    }
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
    int n = 11008, d = 1;
    int batch = 4;
    float *xout, *x, *w, *x_cpu, *x_gpu, *w_gpu, *x_out_gpu;
    CHECK_HIP(hipHostMalloc((void **)&xout, n * BATCH * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipHostMalloc((void **)&x_cpu, n * BATCH * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipHostMalloc((void **)&x, n * BATCH * sizeof(float), hipMemAllocationTypePinned));
    CHECK_HIP(hipHostMalloc((void **)&w, n * sizeof(float), hipMemAllocationTypePinned));

    CHECK_HIP(hipMalloc((void **)&x_gpu, BATCH * n * sizeof(float)));
    CHECK_HIP(hipMalloc((void **)&w_gpu, BATCH * n * sizeof(float)));
    CHECK_HIP(hipMalloc((void **)&x_out_gpu, BATCH * n * sizeof(float)));
    CHECK_HIP(hipDeviceSynchronize());

    // rand_mat(xout,d);
    rand_mat(x, n * BATCH);
    rand_mat(w, n);

    //     for (int i =0;i<n;i++)
    //         printf("x[%d]: %f\n",i,x[i]);
    //  for (int j =0;j<d;j++)
    //     for (int i =0;i<n;i++)

    //         printf("w[%d][%d]: %f\n",j,i,w[j*n+i]);

    CHECK_HIP(hipMemcpy(x_gpu, x, BATCH * n * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(w_gpu, w,  n * sizeof(float), hipMemcpyHostToDevice));
    rmsnorm_cpu(x_cpu, x, w, n);
    int err_count = 0;

    // dim3 blockDim_(BLOCK_SIZE);
    // dim3 gridDim_( 1);

    // printf("total time cost %lf\n", end_ - start_);
    // printf("Gflop: %lf\n", 2.0 * n * d / (end_ - start_) / 1e9);
    rmsnorm_kernel_batch<<<BATCH, 256>>>(xout, x_gpu, w_gpu, n);

    CHECK_HIP(hipDeviceSynchronize());
    // exit(0);
    for (int i = 0; i < BATCH * n; i++)
    {
        if (x_cpu[i] - xout[i] > 1e-5)
        {
            printf("cpu x[%d] %8f but get gpu %8f\n", i, x_cpu[i], xout[i]);
        }
    }

    double start_ = get_time();

    rmsnorm_kernel_batch<<<BATCH, 256>>>(x_out_gpu, x_gpu, w_gpu, n);

    CHECK_HIP(hipDeviceSynchronize());
    double end_ = get_time();

    printf("total time cost %lf\n", end_ - start_);
    printf("Gflop: %lf\n", BATCH * 3.0 * n / (end_ - start_) / 1e9);

    return 0;
}
