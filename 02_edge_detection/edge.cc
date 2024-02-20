#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>

#include <jpeglib.h>
#include <jerror.h>
#include <hip/hip_runtime.h>
using namespace std;
#define BLOCK_SIZE 16
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
int timespec_subtract(struct timespec *result, struct timespec *x, struct timespec *y)
{
  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_nsec < y->tv_nsec)
  {
    int nsec = (y->tv_nsec - x->tv_nsec) / 1000000000 + 1;
    y->tv_nsec -= 1000000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_nsec - y->tv_nsec > 1000000000)
  {
    int nsec = (x->tv_nsec - y->tv_nsec) / 1000000000;
    y->tv_nsec += 1000000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_nsec is certainly positive. */
  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_nsec = x->tv_nsec - y->tv_nsec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}

struct Pixel
{
  unsigned char r, g, b;
};

struct Image
{
  Pixel *pixels;
  int width, height;

  Pixel &getPixel(int x, int y)
  {
    return pixels[y * width + x];
  }

  void copy(Image &out)
  {
    out.width = width;
    out.height = height;
    // out.pixels = (Pixel*)malloc(width*height*sizeof(Pixel));
    // printf("Done mem copy size of Pixel %ld \n", sizeof(Pixel));
    // exit(0);
    // printf("%d %d ",width,height);
    CHECK_HIP(hipHostMalloc((void **)&(out.pixels), width * height * sizeof(Pixel), hipMemAllocationTypePinned));
    memcpy(out.pixels, pixels, width * height * sizeof(Pixel));
  }
};

// Function to read an image as a JPG file
bool readJPEGImage(const string &filename, Image &img)
{
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  JSAMPROW row_pointer;
  FILE *fp;

  if ((fp = fopen(filename.c_str(), "rb")) == NULL)
  {
    fprintf(stderr, "can't open %s\n", filename.c_str());
    return false;
  }

  cinfo.err = jpeg_std_error(&jerr);

  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, fp);

  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);

  if (cinfo.num_components != 3)
  {
    fprintf(stderr, "JPEG file with 3 channels is only supported\n");
    fprintf(stderr, "%s has %d channels\n", filename.c_str(), cinfo.num_components);
    return false;
  }

  img.width = cinfo.output_width;
  img.height = cinfo.output_height;

  // img.pixels = (Pixel *)malloc(img.width * img.height * sizeof(Pixel));
  CHECK_HIP(hipHostMalloc((void **)&img.pixels, img.width * img.height * sizeof(Pixel), hipMemAllocationTypePinned));
  for (int i = 0; i < img.height; i++)
  {
    row_pointer = (JSAMPROW)&img.pixels[i * img.width];
    jpeg_read_scanlines(&cinfo, &row_pointer, 1);
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  fclose(fp);
  return true;
}

// Function to save an image as a JPG file
bool saveJPEGImage(const string &filename, const Image &img)
{
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  JSAMPROW row_pointer;
  FILE *fp;

  cinfo.err = jpeg_std_error(&jerr);

  if ((fp = fopen(filename.c_str(), "wb")) == NULL)
  {
    fprintf(stderr, "can't open %s\n", filename.c_str());
    return false;
  }

  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, fp);

  cinfo.image_width = img.width;
  cinfo.image_height = img.height;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;

  jpeg_set_defaults(&cinfo);
  jpeg_start_compress(&cinfo, TRUE);

  for (int i = 0; i < img.height; i++)
  {
    row_pointer = (JSAMPROW)&img.pixels[i * img.width];
    jpeg_write_scanlines(&cinfo, &row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  fclose(fp);

  return true;
}

// Function to apply the Sobel filter for edge detection
void applySobelFilter(Image &input, Image &output, const int filterX[3][3], const int filterY[3][3])
{
  //   for (int i = -1; i <= 1; ++i)

  //         for (int j = -1; j <= 1; ++j){
  //           printf("(x=%d,y=%d)  %d\n",j+1,i+1,filterX[i + 1][j + 1]);
  //         }
  // printf("--------------\n");
  //   for (int i = -1; i <= 1; ++i)

  //         for (int j = -1; j <= 1; ++j){
  //           printf("(x=%d,y=%d)  %d\n",j+1,i+1,filterY[i + 1][j + 1]);
  //         }

  for (int y = 1; y < input.height - 1; ++y)
  {
    for (int x = 1; x < input.width - 1; ++x)
    {
      int gx = 0, gy = 0;
      for (int i = -1; i <= 1; ++i)
      {
        for (int j = -1; j <= 1; ++j)
        {
          Pixel p = input.getPixel(x + j, y + i);

          gx += ((int)p.r + (int)p.g + (int)p.b) / 3 * filterX[i + 1][j + 1];
          gy += ((int)p.r + (int)p.g + (int)p.b) / 3 * filterY[i + 1][j + 1];
        }
      }
      int magnitude = static_cast<int>(sqrt(gx * gx + gy * gy));
      magnitude = min(max(magnitude, 0), 255);
      output.getPixel(x, y) = {static_cast<unsigned char>(magnitude),
                               static_cast<unsigned char>(magnitude),
                               static_cast<unsigned char>(magnitude)};
    }
  }
}

__global__ void sobel_kernel_native(Pixel *input_pixels, Pixel *output_pixels, int width, int height)
{
  int gray;
  Pixel p;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  // __shared__ int grays[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
  if (x > 0 && y > 0 && x < width - 1 && y < height - 1)
  {
  //   p = input_pixels[x + width * (y)];
  //   gray = ((int)p.r + (int)p.g + (int)p.b) / 3;
  //   grays[1 + threadIdx.y][1 + threadIdx.x] = gray;
  //   if (threadIdx.x == 0 && threadIdx.y == 0)
  //   {
  //     p = input_pixels[x - 1 + width * (y - 1)];
  //     gray = ((int)p.r + (int)p.g + (int)p.b) / 3;
  //     grays[0][0] = gray;
  //   }
  //   else if (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE - 1)
  //   {
  //     p = input_pixels[x - 1 + width * (y + 1)];
  //     gray = ((int)p.r + (int)p.g + (int)p.b) / 3;
  //     grays[BLOCK_SIZE + 1][0] = gray;
  //   }
  //   else if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == 0)
  //   {
  //     p = input_pixels[x + 1 + width * (y - 1)];
  //     gray = ((int)p.r + (int)p.g + (int)p.b) / 3;
  //     grays[0][BLOCK_SIZE + 1] = gray;
  //   }
  //   else if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == BLOCK_SIZE - 1)
  //   {
  //     p = input_pixels[x + 1 + width * (y + 1)];
  //     gray = ((int)p.r + (int)p.g + (int)p.b) / 3;
  //     grays[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = gray;
  //   }

  //   if (threadIdx.x == 0)
  //   {
  //     p = input_pixels[x - 1 + width * (y)];
  //     gray = ((int)p.r + (int)p.g + (int)p.b) / 3;
  //     grays[1 + threadIdx.y][0] = gray;
  //   }
  //   else if (threadIdx.y == 0)
  //   {
  //     p = input_pixels[x + width * (y - 1)];
  //     gray = ((int)p.r + (int)p.g + (int)p.b) / 3;
  //     grays[0][1 + threadIdx.x] = gray;
  //   }
  //   else if (threadIdx.x == BLOCK_SIZE - 1)
  //   {
  //     p = input_pixels[x + 1 + width * (y)];
  //     gray = ((int)p.r + (int)p.g + (int)p.b) / 3;
  //     grays[1 + threadIdx.y][BLOCK_SIZE + 1] = gray;
  //   }
  //   else if (threadIdx.y == BLOCK_SIZE - 1)
  //   {
  //     p = input_pixels[x + width * (y + 1)];
  //     gray = ((int)p.r + (int)p.g + (int)p.b) / 3;
  //     grays[BLOCK_SIZE + 1][1 + threadIdx.x] = gray;
  //   }

  //   __syncthreads();

    // Pixel p00, p01, p02, p10, p11, p12, p20, p21, p22;
    int magnitude;
    // int width = input.width;

    // p00 = input_pixels[y * width + x];
    // p10 = input_pixels[x + 1 + width * y];
    // p20 = input_pixels[x + 2 + width * y];

    // p01 = input_pixels[x + width * (y + 1)];
    // p11 = input_pixels[x + 1 + width * (y + 1)];
    // p21 = input_pixels[x + 2 + width * (y + 1)];

    // p02 = input_pixels[x + width * (y + 2)];
    // p12 = input_pixels[x + 1 + width * (y + 2)];
    // p22 = input_pixels[x + 2 + width * (y + 2)];
    const int filterX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int filterY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // gx = (-1 * (int)(p00.r + p00.g + p00.b) / 3 + (int)(p20.r + p20.g + p20.b) / 3 - 2 * (int)(p01.r + p01.g + p01.b) / 3 + 2 * (int)(p21.r + p21.g + p21.b) / 3 - 1 * (int)(p02.r + p02.g + p02.b) / 3 + (int)(p22.r + p22.g + p22.b) / 3);
    // gy = (-1 * (int)(p00.r + p00.g + p00.b) / 3 - 2 * (int)(p10.r + p10.g + p10.b) / 3 - 1 * (int)(p20.r + p20.g + p20.b) / 3 + (int)(p02.r + p02.g + p02.b) / 3 + 2 * (int)(p12.r + p12.g + p12.b) / 3 + (int)(p22.r + p22.g + p22.b) / 3);
    int gx = 0, gy = 0;
    for (int i = -1; i <= 1; ++i)
    {
      for (int j = -1; j <= 1; ++j)
      {
        p = input_pixels[x + j + width * (y + i)];
        int gray = ((int)p.r + (int)p.g + (int)p.b) / 3;
        // gray = grays[1 + threadIdx.y + i][1 + threadIdx.x + j];
        gx += gray * filterX[i + 1][j + 1];
        gy += gray * filterY[i + 1][j + 1];
      }
    }
    magnitude = static_cast<int>(sqrt(gx * gx + gy * gy));
    magnitude = min(max(magnitude, 0), 255);
    output_pixels[x + width * (y)] = {static_cast<unsigned char>(magnitude),
                                      static_cast<unsigned char>(magnitude),
                                      static_cast<unsigned char>(magnitude)};
  }
}

void applySobelFilter_hip(Image &input, Image &output, int width, int height)
{
  // TODO:
  dim3 blockdim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 griddim((input.width + blockdim.x - 1) / blockdim.x, (input.height + blockdim.y - 1) / blockdim.y);
  sobel_kernel_native<<<griddim, blockdim>>>(input.pixels, output.pixels, width, height);
  CHECK_HIP(hipDeviceSynchronize());
}

int main(int argc, char **argv)
{
  string inputFilename;
  string outputFilename;
  string outputFilename_hip;
  int verify = 0;

  if (argc < 4)
  {
    fprintf(stderr, "$> edge <input_filename> <output_filename_seq> <output_filename_hip> <verification:0|1>\n");
    return 1;
  }
  else
  {
    inputFilename = argv[1];
    outputFilename = argv[2];
    outputFilename_hip = argv[3];
    if (argc > 4)
    {
      verify = atoi(argv[4]);
    }
  }

  const int filterX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  const int filterY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  struct timespec start, end, spent;

  Image inputImage;
  Image outputImage;
  Image outputImage_hip;

  if (!readJPEGImage(inputFilename, inputImage))
  {
    return -1;
  }
  printf("image size: %d %d\n", inputImage.height, inputImage.width);

  inputImage.copy(outputImage);     // Copy input image properties to output image
  inputImage.copy(outputImage_hip); // Copy input image properties to output image

  clock_gettime(CLOCK_MONOTONIC, &start);
  applySobelFilter(inputImage, outputImage, filterX, filterY);
  clock_gettime(CLOCK_MONOTONIC, &end);
  timespec_subtract(&spent, &end, &start);
  printf("CPU Time spent: %ld.%09ld\n", spent.tv_sec, spent.tv_nsec);

  clock_gettime(CLOCK_MONOTONIC, &start);

  // You may modify this code part
  //{
  applySobelFilter_hip(inputImage, outputImage_hip, inputImage.width, inputImage.height);
  //}

  clock_gettime(CLOCK_MONOTONIC, &end);
  timespec_subtract(&spent, &end, &start);
  printf("GPU Time spent: %ld.%09ld\n", spent.tv_sec, spent.tv_nsec);
  //}

  // Save the output image
  saveJPEGImage(outputFilename, outputImage);
  saveJPEGImage(outputFilename_hip, outputImage_hip);

  // verfication (CPU vs GPU)
  if (verify == 1)
  {
    // Verification
    bool pass = true;
    int count = 0;
    for (int i = 0; i < outputImage.width * outputImage.height; i++)
    {
      if (outputImage.pixels[i].r != outputImage_hip.pixels[i].r)
      {
        // printf("[%d] r=%d vs %d : %d\n", i, outputImage.pixels[i].r, outputImage_hip.pixels[i].r, inputImage.pixels[i].r);
        pass = false;
        count++;
      }
      if (outputImage.pixels[i].g != outputImage_hip.pixels[i].g)
      {
        // printf("[%d] g=%d vs %d : %d\n", i, outputImage.pixels[i].g, outputImage_hip.pixels[i].g, inputImage.pixels[i].g);
        pass = false;
        count++;
      }
      if (outputImage.pixels[i].b != outputImage_hip.pixels[i].b)
      {
        // printf("[%d] b=%d vs %d : %d\n", i, outputImage.pixels[i].b, outputImage_hip.pixels[i].b, inputImage.pixels[i].b);
        pass = false;
        count++;
      }
    }
    if (pass)
    {
      printf("Verification Pass!\n");
    }
    else
    {
      printf("Verification Failed! (%d)\n", count);
    }
  }

  CHECK_HIP(hipFree(inputImage.pixels));
  CHECK_HIP(hipFree(outputImage.pixels));
  CHECK_HIP(hipFree(outputImage_hip.pixels));

  return 0;
}
