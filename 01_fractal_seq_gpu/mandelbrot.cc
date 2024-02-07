#include <stdio.h>
#include <stdlib.h>
// #include <math.h>
#include <hip/hip_runtime.h>
#include "timers.h"

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
#define COUNT_MAX 5000
#define BLOCK_SIZE 16
#define MIN(x, y) ((x) < (y) ? (x) : (y))

#ifdef SAVE_JPG
void save_jpeg_image(const char *filename, int *r, int *g, int *b, int image_width, int image_height);
#endif

// void cpu_mandelbrot(int *r, int *g, int *b, int m, int n)
// {
//   int c;
//   int count_max = COUNT_MAX;
//   int i, j, k;
//   float x_max = 1.25;
//   float x_min = -2.25;

//   float x, x1, x2;
//   float y_max = 1.75;
//   float y_min = -1.75;

//   float y, y1, y2;
//   for (i = 0; i < m; i++)
//   {
//     for (j = 0; j < n; j++)
//     {
//       x = ((float)(j - 1) * x_max + (float)(m - j) * x_min) / (float)(m - 1);

//       y = ((float)(i - 1) * y_max + (float)(n - i) * y_min) / (float)(n - 1);

//       int count = 0;

//       x1 = x;
//       y1 = y;

//       for (k = 1; k <= count_max; k++)
//       {
//         x2 = x1 * x1 - y1 * y1 + x;
//         y2 = 2.0 * x1 * y1 + y;

//         if (x2 < -2.0 || 2.0 < x2 || y2 < -2.0 || 2.0 < y2)
//         {
//           count = k;
//           break;
//         }
//         x1 = x2;
//         y1 = y2;
//       }

//       if ((count % 2) == 1)
//       {
//         r[i * n + j] = 255;
//         g[i * n + j] = 255;
//         b[i * n + j] = 255;
//       }
//       else
//       {
//         c = (int)(255.0 * sqrtf(sqrtf(sqrtf(
//                               ((float)(count) / (float)(count_max))))));
//         r[i * n + j] = 3 * c / 5;
//         g[i * n + j] = 3 * c / 5;
//         b[i * n + j] = c;
//       }
//     }
//   }
// }

__global__ void gpu_mandelbrot(int *r, int *g, int *b, int m, int n)
{
  int c;
  int count_max = COUNT_MAX;
  int i, j, k;
  float x_max = 1.25;
  float x_min = -2.25;

  float x, x1, x2;
  float y_max = 1.75;
  float y_min = -1.75;

  float y, y1, y2;

  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= m || j >= n)
    return;
  x = ((float)(j - 1) * x_max + (float)(m - j) * x_min) / (float)(m - 1);

  y = ((float)(i - 1) * y_max + (float)(n - i) * y_min) / (float)(n - 1);

  int count = 0;

  x1 = x;
  y1 = y;

  for (k = 1; k <= count_max; k++)
  {
    x2 = x1 * x1 - y1 * y1 + x;
    y2 = 2.0 * x1 * y1 + y;

    if (x2 < -2.0 || 2.0 < x2 || y2 < -2.0 || 2.0 < y2)
    {
      count = k;
      break;
    }
    x1 = x2;
    y1 = y2;
  }
  int indx = i * n + j;

  if ((count % 2) == 1)
  {
    r[indx] = 255;
    g[indx] = 255;
    b[indx] = 255;
  }
  else
  {
    c = (int)(255.0 * sqrtf(sqrtf(sqrtf(
                          ((float)(count) / (float)(count_max))))));
    r[indx] = 3 * c / 5;
    g[indx] = 3 * c / 5;
    b[indx] = c;
  }
}

void mandelbrot(int m, int n, char *output_filename)
{
  // int c;
  int count_max = COUNT_MAX;
  int i;
  float x_max = 1.25;
  float x_min = -2.25;
  float y_max = 1.75;
  float y_min = -1.75;
#ifndef SAVE_JPG
  int jhi, jlo;
  FILE *output_unit;
#endif
  double wtime;

  // int *r_cpu = (int *)calloc(m * n, sizeof(int));
  // int *g_cpu = (int *)calloc(m * n, sizeof(int));
  // int *b_cpu = (int *)calloc(m * n, sizeof(int));

  int *r_gpu, *g_gpu, *b_gpu, *r, *g, *b;
  printf("Start hip Malloc\n");
  CHECK_HIP(hipMalloc((void **)&r_gpu, m * n * sizeof(int)));
  CHECK_HIP(hipMalloc((void **)&g_gpu, m * n * sizeof(int)));
  CHECK_HIP(hipMalloc((void **)&b_gpu, m * n * sizeof(int)));
  printf("Done hip Malloc\n");

  CHECK_HIP(hipHostMalloc((void **)&r, m * n * sizeof(int)));
  CHECK_HIP(hipHostMalloc((void **)&g, m * n * sizeof(int)));
  CHECK_HIP(hipHostMalloc((void **)&b, m * n * sizeof(int)));

  hipStream_t streams[3];

  for (i = 0; i < 3; i++)
  {
    CHECK_HIP(hipStreamCreate(&streams[i]));
  }

  printf("  Sequential C version\n");
  printf("\n");
  printf("  Create an ASCII PPM image of the Mandelbrot set.\n");
  printf("\n");
  printf("  For each point C = X + i*Y\n");
  printf("  with X range [%g,%g]\n", x_min, x_max);
  printf("  and  Y range [%g,%g]\n", y_min, y_max);
  printf("  carry out %d iterations of the map\n", count_max);
  printf("  Z(n+1) = Z(n)^2 + C.\n");
  printf("  If the iterates stay bounded (norm less than 2)\n");
  printf("  then C is taken to be a member of the set.\n");
  printf("\n");
  printf("  An image of the set is created using\n");
  printf("    M = %d pixels in the X direction and\n", m);
  printf("    N = %d pixels in the Y direction.\n", n);

  timer_init();
  timer_start(0);

  // Carry out the iteration for each pixel, determining COUNT.
  // cpu_mandelbrot(r_cpu, g_cpu, b_cpu, m, n);
  // int *r = r_cpu;
  // int *g = g_cpu;
  // int *b = b_cpu;

  dim3 blockdim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 griddim((m + blockdim.x - 1) / blockdim.x, (n + blockdim.y - 1) / blockdim.y);

  gpu_mandelbrot<<<griddim, blockdim>>>(r_gpu, g_gpu, b_gpu, m, n);
  // CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpyAsync(r, r_gpu, m * n * sizeof(int), hipMemcpyDeviceToHost, streams[0]));
  CHECK_HIP(hipMemcpyAsync(g, g_gpu, m * n * sizeof(int), hipMemcpyDeviceToHost, streams[1]));
  CHECK_HIP(hipMemcpyAsync(b, b_gpu, m * n * sizeof(int), hipMemcpyDeviceToHost, streams[2]));

  CHECK_HIP(hipStreamSynchronize(streams[0]));
  CHECK_HIP(hipStreamSynchronize(streams[1]));
  CHECK_HIP(hipStreamSynchronize(streams[2]));
  CHECK_HIP(hipDeviceSynchronize());

  timer_stop(0);
  wtime = timer_read(0);
  printf("\n");
  printf("  Time = %lf seconds.\n", wtime);

#ifdef SAVE_JPG
  // Write data to an JPEG file.
  save_jpeg_image(output_filename, r, g, b, n, m);
#else
  // Write data to an ASCII PPM file.
  output_unit = fopen(output_filename, "wt");

  fprintf(output_unit, "P3\n");
  fprintf(output_unit, "%d  %d\n", n, m);
  fprintf(output_unit, "%d\n", 255);
  for (i = 0; i < m; i++)
  {
    for (jlo = 0; jlo < n; jlo = jlo + 4)
    {
      jhi = MIN(jlo + 4, n);
      for (j = jlo; j < jhi; j++)
      {
        fprintf(output_unit, "  %d  %d  %d", r[i * n + j], g[i * n + j], b[i * n + j]);
      }
      fprintf(output_unit, "\n");
    }
  }
  fclose(output_unit);
#endif

  printf("\n");
  printf("  Graphics data written to \"%s\".\n\n", output_filename);

  // Terminate.
  CHECK_HIP(hipFree(r));
  CHECK_HIP(hipFree(g));
  CHECK_HIP(hipFree(b));
  CHECK_HIP(hipFree(r_gpu));
  CHECK_HIP(hipFree(g_gpu));
  CHECK_HIP(hipFree(b_gpu));
}
