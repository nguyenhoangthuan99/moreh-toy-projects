#include "kmeans.h"
#include <hip/hip_runtime.h>
#include <stdlib.h>
#include <float.h>
#include <chrono>
#include <iostream>
#define BLOCK_SIZE 128
#define WARP_SIZE 64
#define TILE_SIZE 2
#define MAX_DATA_N 9e6
#define MAX_CLASS_N 1024
#define CHECK_HIP(cmd)                                                                                     \
  do                                                                                                       \
  {                                                                                                        \
    hipError_t error = cmd;                                                                                \
    if (error != hipSuccess)                                                                               \
    {                                                                                                      \
      fprintf(stderr, "HIP Error: %s (%d): %s:%d\n", hipGetErrorString(error), error, __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                                                  \
    }                                                                                                      \
  } while (0)
__device__ int warpReduceSumInt(int val)
{
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    val += __shfl_down(val, offset);
  return val;
}

__device__ int blockReduceSumInt(int val)
{
  static __shared__ int shared[WARP_SIZE];
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;
  val = warpReduceSumInt(val);
  if (lane == 0)
    shared[wid] = val;

  __syncthreads();
  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

  if (wid == 0)
    val = warpReduceSumInt(val);
  return val;
}
__device__ Point warpReduceSumPoint(Point val)
{
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
  {
    val.x += __shfl_down(val.x, offset);
    val.y += __shfl_down(val.y, offset);
  }

  return val;
}

__device__ Point blockReduceSumPoint(Point val)
{
  static __shared__ Point shared[WARP_SIZE];
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;
  val = warpReduceSumPoint(val);
  if (lane == 0)
    shared[wid] = val;

  __syncthreads();
  Point t;
  t.x = 0, t.y = 0;
  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : t;

  if (wid == 0)
    val = warpReduceSumPoint(val);
  return val;
}
__device__ double warpReduceSum(double val)
{
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    val += __shfl_down(val, offset);
  return val;
}
__device__ double warpReduceMin(double val)
{
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    val = min(val, __shfl_down(val, offset));
  return val;
}
__device__ double blockReduceSum(double val)
{
  static __shared__ double shared[WARP_SIZE];
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

// __global__ void assign_cluster_shared(Point *centroids, Point *data, int *partitioned, int class_n, int data_n)
// {
//   __shared__ Point centroids_shared[MAX_CLASS_N + 1];

//   if (threadIdx.x < class_n)
//     centroids_shared[threadIdx.x] = centroids[threadIdx.x];

//   __syncthreads();
//   int block_i = blockIdx.x * blockDim.x + threadIdx.x;

//   double min_dist[TILE_SIZE] = {DBL_MAX};
//   int res[TILE_SIZE];
//   // #pragma unroll 20

//   // Point center = centroids_shared[class_i];

//   for (int tile_i = 0; tile_i < TILE_SIZE; tile_i++)
//   {
//     if (block_i * TILE_SIZE + tile_i < data_n)
//     {
//       Point data_ = data[block_i * TILE_SIZE + tile_i];
//       for (int class_i = 0; class_i < class_n; class_i++)
//       {
//         double x = data_.x - centroids_shared[class_i].x;
//         double y = data_.y - centroids_shared[class_i].y;

//         double dist = x * x + y * y;

//         if (dist < min_dist[tile_i])
//         {
//           res[tile_i] = class_i;
//           min_dist[tile_i] = dist;
//         }
//       }
//     }
//   }
//   // int left = min(class_n - TILE_SIZE * block_i, TILE_SIZE);
//   for (int i = 0; i < TILE_SIZE ; i++)
//   {
//     if (block_i * TILE_SIZE + i < data_n)
//       partitioned[block_i * TILE_SIZE + i] = res[i];
//   }
// }

__global__ void assign_cluster_v2(Point *centroids, Point *data, int *partitioned, int class_n, int data_n)
{
  int data_i = blockIdx.x * blockDim.x + threadIdx.x;
  double min_dist = DBL_MAX;
  int res;
#pragma unroll 32
  for (int class_i = threadIdx.x; class_i < class_n; class_i += blockDim.x)
  {

    double x = data[data_i].x - centroids[class_i].x;
    double y = data[data_i].y - centroids[class_i].y;

    double dist = x * x + y * y;

    if (dist < min_dist)
    {
      res = class_i;
      min_dist = dist;
    }
  }
  double min_dist_global = warpReduceMin(min_dist);
  if (min_dist == min_dist_global)
    partitioned[data_i] = res;
  // printf("Done thread %d\n",data_i);
}
__global__ void assign_cluster(Point *centroids, Point *data, int *count, int *partitioned, int class_n, int data_n, int data_offset = 0, int max_data = MAX_DATA_N)
{
  // __shared__ Point centroid_shared[MAX_CLASS_N + 1];

  int data_i = data_offset + blockIdx.x * blockDim.x + threadIdx.x;
  double min_dist = DBL_MAX;
  int res;
  // if (threadIdx.x < class_n)
    // centroid_shared[threadIdx.x] = centroids[threadIdx.x];
  // __syncthreads();
  if (data_i >= min(data_n, max_data))
    return;
#pragma unroll 24
  for (int class_i = 0; class_i < class_n; class_i++)
  {

    double x = data[data_i].x - centroids[class_i].x;
    double y = data[data_i].y - centroids[class_i].y;

    double dist = x * x + y * y;

    if (dist < min_dist)
    {
      res = class_i;
      min_dist = dist;
    }
  }
  partitioned[data_i] = res;
}
// __global__ void sum_and_count(Point *centroids, Point *data, int *partitioned, int *count, int class_n, int data_n)
// {
//   int data_i = blockIdx.x * blockDim.x + threadIdx.x;

//   atomicAdd(&centroids[partitioned[data_i]].x, data[data_i].x);
//   atomicAdd(&centroids[partitioned[data_i]].y, data[data_i].y);
//   atomicAdd(&count[partitioned[data_i]], 1);
// }

__global__ void sum_and_count_v2(Point *centroids, Point *data, int *count, int *partitioned, int class_n, int data_n)
{
  int class_i = blockIdx.x;
  if (class_i >= class_n)
    return;
  // centroids[class_i].x = 0;
  // centroids[class_i].y = 0;
  // count[class_i] = 0;
  // int data_i = blockIdx.x * blockDim.x + threadIdx.x;
  Point t;
  t.x = 0, t.y = 0;
  int count_i = 0;
  for (int data_i = threadIdx.x; data_i < data_n; data_i += blockDim.x)
  {
    if (partitioned[data_i] == class_i)
    {
      t.x += data[data_i].x;
      t.y += data[data_i].y;
      count_i += 1;
    }
  }
  // t.x = blockReduceSum(t.x);
  // t.y = blockReduceSum(t.y);
  t = blockReduceSumPoint(t);
  count_i = blockReduceSumInt(count_i);
  if (threadIdx.x == 0)
  {
    centroids[class_i].x = t.x / count_i;
    centroids[class_i].y = t.y / count_i;
    // count[class_i] = count_i;
  }
}
__global__ void sum_and_count_v3(Point *centroids, Point *data, int *count, int *partitioned, int class_n, int data_n, int class_offset = 0, int max_class = 128)
{
  int class_i = class_offset + blockIdx.x;
  int part_i = blockIdx.y;
  int num_part = gridDim.y;
  if (class_i >= min(class_n, max_class))
    return;
  Point t;
  t.x = 0, t.y = 0;
  int count_i = 0;

  int start = part_i * data_n / num_part;                                       // max(0, (int)data_n / num_part * part_i + min(part_i, (int)data_n % num_part) - 1);
  int end = part_i == num_part - 1 ? data_n : (part_i + 1) * data_n / num_part; //(int)data_n / num_part * (part_i + 1) + std::min(part_i + 1, (int)data_n % num_part);
  // if (threadIdx.x == 0)
  //   printf("part %d, start %d, end %d\n", part_i, start, end);
  for (int data_i = start + threadIdx.x; data_i < end; data_i += blockDim.x)
  {
    if (partitioned[data_i] == class_i)
    {
      t.x += data[data_i].x;
      t.y += data[data_i].y;
      count_i += 1;
    }
  }
  // t.x = blockReduceSum(t.x);
  // t.y = blockReduceSum(t.y);
  t = blockReduceSumPoint(t);
  count_i = blockReduceSumInt(count_i);
  if (threadIdx.x == 0)
  {
    atomicAdd(&centroids[class_i].x, t.x);
    atomicAdd(&centroids[class_i].y, t.y);
    atomicAdd(&count[class_i], count_i);
  }
  // if (threadIdx.x == 0)
  // {
  //   centroids[class_i].x = t.x ;
  //   centroids[class_i].y = t.y ;
  //   count[class_i] = count_i;
  // }
}
__global__ void set_zeros(Point *centroids, int *count, int class_n, int class_offset = 0, int max_class = MAX_CLASS_N)
{
  int class_i = class_offset + blockIdx.x * blockDim.x + threadIdx.x;
  if (class_i >= min(class_n, max_class))
    return;
  centroids[class_i].x = 0.0;
  centroids[class_i].y = 0.0;
  count[class_i] = 0;
}
__global__ void calulate_centroids(Point *centroids, int *count, int class_n, int class_offset = 0, int max_class = MAX_CLASS_N)
{
  int class_i = class_offset + blockIdx.x * blockDim.x + threadIdx.x;
  if (class_i >= min(class_n, max_class))
    return;
  centroids[class_i].x /= count[class_i];
  centroids[class_i].y /= count[class_i];
}

__global__ void set_zeros_all(Point *centroids, int *count, int class_n, int class_offset = 0, int max_class = MAX_CLASS_N)
{
  for (int class_i = class_offset; class_i < min(class_n, max_class); class_i++)
  {
    centroids[class_i].x = 0.0;
    centroids[class_i].y = 0.0;
    count[class_i] = 0;
  }
}
__global__ void calulate_centroids_all(Point *centroids, int *count, int class_n, int class_offset = 0, int max_class = MAX_CLASS_N)
{
  for (int class_i = class_offset; class_i < min(class_n, max_class); class_i++)
  {
    centroids[class_i].x /= count[class_i];
    centroids[class_i].y /= count[class_i];
  }
}

void run_1_gpu(int iteration_n, int class_n, int data_n, Point *centroids, Point *data, int *partitioned)
{
  int *count;
  Point *centroids_gpu;
  Point *data_gpu;
  int *partitioned_gpu;
  int *flag_gpu;

  // auto start = std::chrono::high_resolution_clock::now();
  CHECK_HIP(hipMallocAsync((void **)&data_gpu, sizeof(Point) * data_n, hipStreamDefault));
  CHECK_HIP(hipMemcpyAsync(data_gpu, data, sizeof(Point) * data_n, hipMemcpyHostToDevice, hipStreamDefault));
  CHECK_HIP(hipMallocAsync((void **)&centroids_gpu, sizeof(Point) * class_n, hipStreamDefault)); //, hipMemAllocationTypePinned));
  CHECK_HIP(hipMemcpyAsync(centroids_gpu, centroids, sizeof(Point) * class_n, hipMemcpyHostToDevice, hipStreamDefault));
  CHECK_HIP(hipMallocAsync((void **)&partitioned_gpu, sizeof(int) * data_n, hipStreamDefault));
  CHECK_HIP(hipMallocAsync((void **)&count, sizeof(int) * class_n, hipStreamDefault));
  // CHECK_HIP(hipMallocAsync((void **)&flag_gpu, sizeof(int), hipStreamDefault));

  // CHECK_HIP(hipHostMalloc((void **)&count, sizeof(int) * class_n, hipMemAllocationTypePinned));
  // CHECK_HIP(hipHostMalloc((void **)&centroids_gpu, sizeof(Point) * class_n, hipMemAllocationTypePinned));
  // CHECK_HIP(hipHostMalloc((void **)&partitioned_gpu, sizeof(int) * data_n, hipMemAllocationTypePinned));
  // CHECK_HIP(hipHostMalloc((void **)&data_gpu, sizeof(Point) * data_n, hipMemAllocationTypePinned));
  // CHECK_HIP(hipMemcpyAsync(data_gpu, data, sizeof(Point) * data_n, hipMemcpyHostToDevice, hipStreamDefault));
  // CHECK_HIP(hipMemcpyAsync(centroids_gpu, centroids, sizeof(Point) * class_n, hipMemcpyHostToDevice, hipStreamDefault));
  // CHECK_HIP(hipHostMalloc((void **)&flag_gpu, sizeof(int), hipMemAllocationTypePinned));
  // CHECK_HIP(hipDeviceSynchronize());
  // auto stop = std::chrono::high_resolution_clock::now();
  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  // std::cout << "Time taken by allocation: " << duration.count() / 1e6 << " seconds" << std::endl;
  int block_size = 1024;
  dim3 blockdim(block_size);
  dim3 griddim(((data_n) + block_size - 1) / block_size);
  // dim3 griddim((data_n/TILE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
  for (int i = 0; i < iteration_n; i++)
  {
    // printf("iteration %d\n", i);
    assign_cluster<<<griddim, blockdim>>>(centroids_gpu, data_gpu, count, partitioned_gpu, class_n, data_n);
    // CHECK_HIP(hipMemset(centroids_gpu, 0, class_n * sizeof(Point)));
    set_zeros<<<(class_n + 63) / 64, 64>>>(centroids_gpu, count, class_n);

    // sum_and_count_v2<<<class_n, 1024>>>(centroids_gpu, data_gpu,count, partitioned_gpu, class_n, data_n);

    dim3 griddim_2(class_n, 32);
    sum_and_count_v3<<<griddim_2, 256>>>(centroids_gpu, data_gpu, count, partitioned_gpu, class_n, data_n);
    calulate_centroids<<<(class_n + 63) / 64, 64>>>(centroids_gpu, count, class_n);
  }
  CHECK_HIP(hipMemcpyAsync(centroids, centroids_gpu, sizeof(Point) * class_n, hipMemcpyDeviceToHost, hipStreamDefault));
  CHECK_HIP(hipMemcpyAsync(partitioned, partitioned_gpu, sizeof(int) * data_n, hipMemcpyDeviceToHost, hipStreamDefault));
  CHECK_HIP(hipDeviceSynchronize());
}
void run_2_gpu(int iteration_n, int class_n, int data_n, Point *centroids, Point *data, int *partitioned)
{
  int *count[2];
  Point *centroids_gpu[2];
  Point *data_gpu[2];
  int *partitioned_gpu[2];

  // auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 2; i++)
  {
    CHECK_HIP(hipSetDevice(i));
    CHECK_HIP(hipMallocAsync((void **)&data_gpu[i], sizeof(Point) * data_n, hipStreamDefault));
    CHECK_HIP(hipMemcpyAsync(data_gpu[i], data, sizeof(Point) * data_n, hipMemcpyHostToDevice, hipStreamDefault));
    CHECK_HIP(hipMallocAsync((void **)&centroids_gpu[i], sizeof(Point) * class_n, hipStreamDefault)); //, hipMemAllocationTypePinned));
    CHECK_HIP(hipMemcpyAsync(centroids_gpu[i], centroids, sizeof(Point) * class_n, hipMemcpyHostToDevice, hipStreamDefault));
    CHECK_HIP(hipMallocAsync((void **)&partitioned_gpu[i], sizeof(int) * data_n, hipStreamDefault));
    CHECK_HIP(hipMallocAsync((void **)&count[i], sizeof(int) * class_n, hipStreamDefault));
  }
  // CHECK_HIP(hipHostMalloc((void **)&centroids_gpu, sizeof(Point) * class_n, hipMemAllocationTypePinned)); //, hipMemAllocationTypePinned));
  // CHECK_HIP(hipMemcpyAsync(centroids_gpu, centroids, sizeof(Point) * class_n, hipMemcpyHostToDevice, hipStreamDefault));
  // CHECK_HIP(hipHostMalloc((void **)&partitioned_gpu, sizeof(int) * data_n, hipMemAllocationTypePinned));
  // CHECK_HIP(hipHostMalloc((void **)&count, sizeof(int) * class_n, hipMemAllocationTypePinned));
  // CHECK_HIP(hipMallocAsync((void **)&flag_gpu, sizeof(int), hipStreamDefault));
  int block_size = 1024;

  // dim3 griddim((data_n/TILE_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
  int classBegin[2];
  int classEnd[2];
  int dataBegin[2];
  int dataEnd[2];
  for (int i = 0; i < 2; i++)
  {
    classBegin[i] = i * class_n / 2;
    classEnd[i] = i == 1 ? class_n : (i + 1) * class_n / 2;
    dataBegin[i] = i * data_n / 2;
    dataEnd[i] = i == 1 ? data_n : (i + 1) * data_n / 2;
  }

  for (int j = 0; j < iteration_n; j++)
  {
    // printf("iteration %d\n", i);
    for (int i = 0; i < 2; i++)
    {
      dim3 blockdim(block_size);
      dim3 griddim(((data_n + 1) / 2 + block_size - 1) / block_size);
      CHECK_HIP(hipSetDevice(i));
      assign_cluster<<<griddim, blockdim>>>(centroids_gpu[i], data_gpu[i], count[i], partitioned_gpu[i], class_n, data_n, dataBegin[i], dataEnd[i]);
      // CHECK_HIP(hipMemset(centroids_gpu, 0, class_n * sizeof(Point)));
      set_zeros_all<<<1, 1>>>(centroids_gpu[i], count[i], class_n, classBegin[i], classEnd[i]);
      int dst = i == 0 ? 1 : 0;
      // CHECK_HIP(hipSetDevice(dst));
      CHECK_HIP(hipMemcpyPeerAsync(partitioned_gpu[dst] + dataBegin[i], dst, partitioned_gpu[i] + dataBegin[i], i, sizeof(int) * (dataEnd[i] - dataBegin[i])));
    }
    for (int i = 0; i < 2; i++)
    {
      CHECK_HIP(hipSetDevice(i));
      CHECK_HIP(hipDeviceSynchronize());
    }

    // set_zeros_all<<<1, 1>>>(centroids_gpu, count, class_n);
    // for (int i = 0; i < 2; i++)
    // {
    //   CHECK_HIP(hipSetDevice(i));
    //   CHECK_HIP(hipDeviceSynchronize());
    // }

    for (int i = 0; i < 2; i++)
    {
      CHECK_HIP(hipSetDevice(i));
      dim3 griddim_2((class_n + 1) / 2, 32);
      sum_and_count_v3<<<griddim_2, 256>>>(centroids_gpu[i], data_gpu[i], count[i], partitioned_gpu[i], class_n, data_n, classBegin[i], classEnd[i]);
      calulate_centroids_all<<<1, 1>>>(centroids_gpu[i], count[i], class_n, classBegin[i], classEnd[i]);
      int dst = i == 0 ? 1 : 0;
      // CHECK_HIP(hipSetDevice(dst));
      CHECK_HIP(hipMemcpyPeerAsync(centroids_gpu[dst] + classBegin[i], dst, centroids_gpu[i] + classBegin[i], i, sizeof(Point) * (classEnd[i] - classBegin[i])));
    }
    for (int i = 0; i < 2; i++)
    {
      CHECK_HIP(hipSetDevice(i));
      CHECK_HIP(hipDeviceSynchronize());
    }
    // calulate_centroids_all<<<1, 1>>>(centroids_gpu, count, class_n);
    // for (int i = 0; i < 2; i++)
    // {
    //   CHECK_HIP(hipSetDevice(i));
    //   CHECK_HIP(hipDeviceSynchronize());
    // }
  }
  // CHECK_HIP(hipSetDevice(i));
  CHECK_HIP(hipMemcpyAsync(centroids, centroids_gpu[1], sizeof(Point) * class_n, hipMemcpyDefault, hipStreamDefault));
  CHECK_HIP(hipMemcpyAsync(partitioned, partitioned_gpu[1], sizeof(int) * data_n, hipMemcpyDefault, hipStreamDefault));
  CHECK_HIP(hipDeviceSynchronize());
}
void kmeans(int iteration_n, int class_n, int data_n, Point *centroids, Point *data, int *partitioned)
{
  // printf("Max memory space: %lf GB %d\n", MAX_DATA_N * MAX_CLASS_N * sizeof(Point) / 1e9, sizeof(Point));
  int ngpu;
  CHECK_HIP(hipGetDeviceCount(&ngpu));

  bool use_2_gpu = ngpu > 1;
  if (use_2_gpu)
  {
    int canAccessPeer;
    CHECK_HIP(hipDeviceCanAccessPeer(&canAccessPeer, 0, 1));
    if (canAccessPeer == 1)
    {
      CHECK_HIP(hipSetDevice(0));
      CHECK_HIP(hipDeviceEnablePeerAccess(1, 0));
    }
    else
    {
      printf("Cannot peer access from %d to %d\n", 0, 1);
    }

    CHECK_HIP(hipDeviceCanAccessPeer(&canAccessPeer, 1, 0));
    if (canAccessPeer == 1)
    {
      CHECK_HIP(hipSetDevice(1));
      CHECK_HIP(hipDeviceEnablePeerAccess(0, 0));
    }
    else
    {
      printf("Cannot peer access from %d to %d\n", 1, 0);
    }
    run_2_gpu(iteration_n, class_n, data_n, centroids, data, partitioned);
  }
  else
  {
    run_1_gpu(iteration_n, class_n, data_n, centroids, data, partitioned);
  }
}
