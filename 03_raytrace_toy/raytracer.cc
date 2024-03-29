// A very basic raytracer example.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>
#include <cassert>
#include <omp.h>

#include <jpeglib.h>
#include <hip/hip_runtime.h>
#ifndef M_PI
#define M_PI 3.141592653589793
#endif

#ifndef INFINITY
#define INFINITY 1e8
#endif
#define BLOCK_SIZE_X 64
#define BLOCK_SIZE_Y 4
#define BUFFER_SIZE 64
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

template <typename T>
class Vec3
{
public:
  T x, y, z;
  Vec3() : x(T(0)), y(T(0)), z(T(0)) {}
  Vec3(T xx) : x(xx), y(xx), z(xx) {}
  Vec3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {}
  Vec3 &normalize()
  {
    T nor2 = length2();
    if (nor2 > 0)
    {
      T invNor = 1 / sqrt(nor2);
      x *= invNor, y *= invNor, z *= invNor;
    }
    return *this;
  }
  Vec3<T> operator*(const T &f) const { return Vec3<T>(x * f, y * f, z * f); }
  Vec3<T> operator*(const Vec3<T> &v) const { return Vec3<T>(x * v.x, y * v.y, z * v.z); }
  T dot(const Vec3<T> &v) const { return x * v.x + y * v.y + z * v.z; }
  Vec3<T> operator-(const Vec3<T> &v) const { return Vec3<T>(x - v.x, y - v.y, z - v.z); }
  Vec3<T> operator+(const Vec3<T> &v) const { return Vec3<T>(x + v.x, y + v.y, z + v.z); }
  Vec3<T> &operator+=(const Vec3<T> &v)
  {
    x += v.x, y += v.y, z += v.z;
    return *this;
  }
  Vec3<T> &operator*=(const Vec3<T> &v)
  {
    x *= v.x, y *= v.y, z *= v.z;
    return *this;
  }
  Vec3<T> operator-() const { return Vec3<T>(-x, -y, -z); }
  T length2() const { return x * x + y * y + z * z; }
  T length() const { return sqrt(length2()); }
  friend std::ostream &operator<<(std::ostream &os, const Vec3<T> &v)
  {
    os << "[" << v.x << " " << v.y << " " << v.z << "]";
    return os;
  }
};

typedef Vec3<float> Vec3f;

class Sphere
{
public:
  Vec3f center;                      /// position of the sphere
  float radius, radius2;             /// sphere radius and radius^2
  Vec3f surfaceColor, emissionColor; /// surface color and emission (light)
  float transparency, reflection;    /// surface transparency and reflectivity
  Sphere(
      const Vec3f &c,
      const float &r,
      const Vec3f &sc,
      const float &refl = 0,
      const float &transp = 0,
      const Vec3f &ec = 0) : center(c), radius(r), radius2(r * r), surfaceColor(sc), emissionColor(ec),
                             transparency(transp), reflection(refl)
  { /* empty */
  }

  // Compute a ray-sphere intersection using the geometric solution
  bool intersect(const Vec3f &rayorig, const Vec3f &raydir, float &t0, float &t1) const
  {
    Vec3f l = center - rayorig;
    float tca = l.dot(raydir);
    if (tca < 0)
      return false;
    float d2 = l.dot(l) - tca * tca;
    if (d2 > radius2)
      return false;
    float thc = sqrt(radius2 - d2);
    t0 = tca - thc;
    t1 = tca + thc;

    return true;
  }
};

template <typename T>
class Vec3GPU
{
public:
  T x, y, z;
  __device__ Vec3GPU() : x(T(0)), y(T(0)), z(T(0)) {}
  __device__ Vec3GPU(T xx) : x(xx), y(xx), z(xx) {}
  __device__ Vec3GPU(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {}
  __device__ Vec3GPU &normalize()
  {
    T nor2 = length();
    if (nor2 > 0)
    {
      // T invNor = 1 / __fsqrt_rn(nor2);
      // x = __fdiv_rn(x, nor2), y = __fdiv_rn(y, nor2), z = __fdiv_rn(z, nor2);
      x = x / nor2, y = y / nor2, z = z / nor2;
    }
    return *this;
  }
  __device__ Vec3GPU<T> operator*(const T &f) const { return Vec3GPU<T>(x * f, y * f, z * f); }
  __device__ Vec3GPU<T> operator*(const Vec3GPU<T> &v) const { return Vec3GPU<T>(x * v.x, y * v.y, z * v.z); }
  __device__ T dot(const Vec3GPU<T> &v) const { return x * v.x + y * v.y + z * v.z; }
  __device__ Vec3GPU<T> operator-(const Vec3GPU<T> &v) const { return Vec3GPU<T>(x - v.x, y - v.y, z - v.z); }
  __device__ Vec3GPU<T> operator+(const Vec3GPU<T> &v) const { return Vec3GPU<T>(x + v.x, y + v.y, z + v.z); }
  __device__ Vec3GPU<T> &operator+=(const Vec3GPU<T> &v)
  {
    x += v.x, y += v.y, z += v.z;
    return *this;
  }
  __device__ Vec3GPU<T> &operator*=(const Vec3GPU<T> &v)
  {
    x *= v.x, y *= v.y, z *= v.z;
    return *this;
  }
  __device__ Vec3GPU<T> operator-() const { return Vec3GPU<T>(-x, -y, -z); }
  __device__ T length2() const { return x * x + y * y + z * z; }
  __device__ T length() const { return __fsqrt_rn(length2()); }
  // __device__ friend std::ostream &operator<<(std::ostream &os, const Vec3GPU<T> &v)
  // {
  //   os << "[" << v.x << " " << v.y << " " << v.z << "]";
  //   return os;
  // }
};

typedef Vec3GPU<float> Vec3fGPU;

class SphereGPU
{
public:
  Vec3fGPU center;                      /// position of the sphere
  float radius, radius2;                /// sphere radius and radius^2
  Vec3fGPU surfaceColor, emissionColor; /// surface color and emission (light)
  float transparency, reflection;       /// surface transparency and reflectivity
  __device__ SphereGPU(
      const Vec3fGPU &c,
      const float &r,
      const Vec3fGPU &sc,
      const float &refl = 0,
      const float &transp = 0,
      const Vec3fGPU &ec = 0) : center(c), radius(r), radius2((r * r)), surfaceColor(sc), emissionColor(ec),
                                transparency(transp), reflection(refl)
  { /* empty */
  }

  // Compute a ray-sphere intersection using the geometric solution
  __device__ bool intersect(const Vec3fGPU &rayorig, const Vec3fGPU &raydir, float &t0, float &t1) const
  {
    Vec3fGPU l = center - rayorig;
    float tca = l.dot(raydir);
    if (tca < 0)
      return false;
    float d2 = l.dot(l) - tca * tca;
    if (d2 > radius2)
      return false;
    float thc = __fsqrt_rn(radius2 - d2);
    t0 = tca - thc;
    t1 = tca + thc;

    return true;
  }
};

// This variable controls the maximum recursion depth
#define MAX_RAY_DEPTH 5

float mix(const float &a, const float &b, const float &mix)
{
  return b * mix + a * (1 - mix);
}

__device__ float mix_gpu(const float &a, const float &b, const float &mix)
{
  return b * mix + a * (1 - mix); //__fadd_rn(__fmul_rn(b, mix), __fmul_rn(a, __fsub_rn(1, mix)));
}
// This is the main trace function. It takes a ray as argument (defined by its origin
// and direction). We test if this ray intersects any of the geometry in the scene.
// If the ray intersects an object, we compute the intersection point, the normal
// at the intersection point, and shade this point using this information.
// Shading depends on the surface property (is it transparent, reflective, diffuse).
// The function returns a color for the ray. If the ray intersects an object that
// is the color of the object at the intersection point, otherwise it returns
// the background color.
Vec3f trace(
    const Vec3f &rayorig,
    const Vec3f &raydir,
    const std::vector<Sphere> &spheres,
    const int &depth)
{
  // if (raydir.length() != 1) std::cerr << "Error " << raydir << std::endl;
  float tnear = INFINITY;
  const Sphere *sphere = NULL;
  // find intersection of this ray with the sphere in the scene
  for (unsigned i = 0; i < spheres.size(); ++i)
  {
    float t0 = INFINITY, t1 = INFINITY;
    if (spheres[i].intersect(rayorig, raydir, t0, t1))
    {
      if (t0 < 0)
        t0 = t1;
      if (t0 < tnear)
      {
        tnear = t0;
        sphere = &spheres[i];
      }
    }
  }
  // if there's no intersection return black or background color
  if (!sphere)
    return Vec3f(2);
  Vec3f surfaceColor = 0;                // color of the ray/surfaceof the object intersected by the ray
  Vec3f phit = rayorig + raydir * tnear; // point of intersection
  Vec3f nhit = phit - sphere->center;    // normal at the intersection point
  nhit.normalize();                      // normalize normal direction
  // If the normal and the view direction are not opposite to each other
  // reverse the normal direction. That also means we are inside the sphere so set
  // the inside bool to true. Finally reverse the sign of IdotN which we want
  // positive.
  float bias = 1e-4; // add some bias to the point from which we will be tracing
  bool inside = false;
  if (raydir.dot(nhit) > 0)
    nhit = -nhit, inside = true;
  if ((sphere->transparency > 0 || sphere->reflection > 0) && depth < MAX_RAY_DEPTH)
  {
    float facingratio = -raydir.dot(nhit);
    // change the mix value to tweak the effect
    float fresneleffect = mix(pow(1 - facingratio, 3), 1, 0.1);
    // compute reflection direction (not need to normalize because all vectors
    // are already normalized)
    Vec3f refldir = raydir - nhit * 2 * raydir.dot(nhit);
    refldir.normalize();
    Vec3f reflection = trace(phit + nhit * bias, refldir, spheres, depth + 1);
    Vec3f refraction = 0;
    // if the sphere is also transparent compute refraction ray (transmission)
    if (sphere->transparency)
    {
      float ior = 1.1, eta = (inside) ? ior : 1 / ior; // are we inside or outside the surface?
      float cosi = -nhit.dot(raydir);
      float k = 1 - eta * eta * (1 - cosi * cosi);
      Vec3f refrdir = raydir * eta + nhit * (eta * cosi - sqrt(k));
      refrdir.normalize();
      refraction = trace(phit - nhit * bias, refrdir, spheres, depth + 1);
    }
    // the result is a mix of reflection and refraction (if the sphere is transparent)
    surfaceColor = (reflection * fresneleffect +
                    refraction * (1 - fresneleffect) * sphere->transparency) *
                   sphere->surfaceColor;
  }
  else
  {
    // it's a diffuse object, no need to raytrace any further
    for (unsigned i = 0; i < spheres.size(); ++i)
    {
      if (spheres[i].emissionColor.x > 0)
      {
        // this is a light
        Vec3f transmission = 1;
        Vec3f lightDirection = spheres[i].center - phit;
        lightDirection.normalize();
        for (unsigned j = 0; j < spheres.size(); ++j)
        {
          if (i != j)
          {
            float t0, t1;
            if (spheres[j].intersect(phit + nhit * bias, lightDirection, t0, t1))
            {
              transmission = 0;
              break;
            }
          }
        }
        surfaceColor += sphere->surfaceColor * transmission *
                        std::max(float(0), nhit.dot(lightDirection)) * spheres[i].emissionColor;
      }
    }
  }

  return surfaceColor + sphere->emissionColor;
}

// Main rendering function. We compute a camera ray for each pixel of the image
// trace it and return a color. If the ray hits a sphere, we return the color of the
// sphere at the intersection point, else we return the background color.
Vec3f *render_cpu(const std::vector<Sphere> &spheres, size_t width, size_t height)
{
  Vec3f *image = new Vec3f[width * height], *pixel = image;
  float invWidth = 1 / float(width), invHeight = 1 / float(height);
  float fov = 30, aspectratio = width / float(height);
  float angle = tan(M_PI * 0.5 * fov / 180.);
  // Trace rays
  // #pragma omp parallel for num_threads(32)
  for (unsigned y = 0; y < height; ++y)
  {
    for (unsigned x = 0; x < width; ++x, ++pixel)
    {
      float xx = (2 * ((x + 0.5) * invWidth) - 1) * angle * aspectratio;
      float yy = (1 - 2 * ((y + 0.5) * invHeight)) * angle;
      Vec3f raydir(xx, yy, -1);
      raydir.normalize();
      *pixel = trace(Vec3f(0), raydir, spheres, 0);
    }
  }

  return image;
}
__device__ Vec3fGPU trace_gpu(
    const Vec3fGPU &rayorig,
    const Vec3fGPU &raydir,
    SphereGPU *spheres,
    int size_spheres,
    const int &depth)
{
  // if (raydir.length() != 1) std::cerr << "Error " << raydir << std::endl;
  float tnear = INFINITY;
  const SphereGPU *sphere = NULL;
  // find intersection of this ray with the sphere in the scene
  for (unsigned i = 0; i < size_spheres; ++i)
  {
    float t0 = INFINITY, t1 = INFINITY;
    if (spheres[i].intersect(rayorig, raydir, t0, t1))
    {
      if (t0 < 0)
        t0 = t1;
      if (t0 < tnear)
      {
        tnear = t0;
        sphere = &spheres[i];
      }
    }
  }
  // if there's no intersection return black or background color
  if (!sphere)
    return Vec3fGPU(2);
  Vec3fGPU surfaceColor = 0;                // color of the ray/surfaceof the object intersected by the ray
  Vec3fGPU phit = rayorig + raydir * tnear; // point of intersection
  Vec3fGPU nhit = phit - sphere->center;    // normal at the intersection point
  nhit.normalize();                         // normalize normal direction
  // If the normal and the view direction are not opposite to each other
  // reverse the normal direction. That also means we are inside the sphere so set
  // the inside bool to true. Finally reverse the sign of IdotN which we want
  // positive.
  float bias = 1e-4; // add some bias to the point from which we will be tracing
  bool inside = false;
  if (raydir.dot(nhit) > 0)
    nhit = -nhit, inside = true;
  Vec3fGPU nhit_bias = nhit * bias;
  Vec3fGPU phit_nhit_bias = phit + nhit_bias;
  if ((sphere->transparency > 0 || sphere->reflection > 0) && depth < MAX_RAY_DEPTH)
  {
    // float facingratio = -raydir.dot(nhit);
    // // change the mix value to tweak the effect
    // float fresneleffect = mix_gpu(pow(1 - facingratio, 3), 1, 0.1);

    float temp = 1 + raydir.dot(nhit);
    float fresneleffect = mix_gpu(temp * temp * temp, 1, 0.1);
    // compute reflection direction (not need to normalize because all vectors
    // are already normalized)
    Vec3fGPU temp_refldir = nhit * raydir.dot(nhit);
    Vec3fGPU refldir = raydir - temp_refldir - temp_refldir;
    // Vec3fGPU refldir = raydir - nhit * 2 * raydir.dot(nhit);
    refldir.normalize();
    Vec3fGPU reflection = trace_gpu(phit_nhit_bias, refldir, spheres, size_spheres, depth + 1);
    Vec3fGPU refraction = 0;
    // if the sphere is also transparent compute refraction ray (transmission)
    if (sphere->transparency)
    {
      float ior = 1.1, eta = (inside) ? ior : 1 / ior; // are we inside or outside the surface?
      float cosi = -nhit.dot(raydir);
      float k = 1 - eta * eta * (1 - cosi * cosi);
      Vec3fGPU refrdir = raydir * eta + nhit * (eta * cosi - __fsqrt_rn(k));
      refrdir.normalize();
      refraction = trace_gpu(phit - nhit_bias, refrdir, spheres, size_spheres, depth + 1);
    }
    // the result is a mix of reflection and refraction (if the sphere is transparent)
    surfaceColor = (reflection * fresneleffect +
                    refraction * (1 - fresneleffect) * sphere->transparency) *
                   sphere->surfaceColor;
  }
  else
  {
    // it's a diffuse object, no need to raytrace any further
    for (unsigned i = 0; i < size_spheres; ++i)
    {
      if (spheres[i].emissionColor.x > 0)
      {
        // this is a light
        Vec3fGPU transmission = 1;
        Vec3fGPU lightDirection = spheres[i].center - phit;
        lightDirection.normalize();
        for (unsigned j = 0; j < size_spheres; ++j)
        {
          if (i != j)
          {
            float t0, t1;
            if (spheres[j].intersect(phit_nhit_bias, lightDirection, t0, t1))
            {
              transmission = 0;
              break;
            }
          }
        }
        surfaceColor += sphere->surfaceColor * transmission *
                        std::max(float(0), nhit.dot(lightDirection)) * spheres[i].emissionColor;
      }
    }
  }

  return surfaceColor + sphere->emissionColor;
}

__global__ void raytrace_kernel(Vec3fGPU *image, size_t width, size_t height, SphereGPU *spheres, int size_spheres, float invWidth, float invHeight, float angle, float aspectratio, int hoffset, int h_work_size)
{
  size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= h_work_size)
    return;
  y += hoffset;
  float temp_xx = ((x + 0.5) * invWidth);
  float xx = (temp_xx + temp_xx - 1) * angle * aspectratio;
  float temp_yy = ((y + 0.5) * invHeight);
  float yy = (1 - temp_yy - temp_yy) * angle;

  Vec3fGPU raydir(xx, yy, -1);
  raydir.normalize();

  image[(y - hoffset) * width + x] = trace_gpu(Vec3fGPU(0), raydir, spheres, size_spheres, 0);
  // printf("%d %d\n",x,y);
}
void mallocSphere()
{
}
Vec3fGPU *render_gpu(const std::vector<Sphere> &spheres, size_t width, size_t height)
{
  // TODO:
  SphereGPU *sphere_gpu[2];
  Vec3fGPU *image;
  // hipStream_t streams[2];
  int ngpu;
  CHECK_HIP(hipGetDeviceCount(&ngpu));
  // printf("num GPUs: %d\n", ngpu);
  int hbegin[2], hend[2];
  for (int i = 0; i < ngpu; i++)
  {

    hbegin[i] = std::max(0, (int)height / ngpu * i + std::min(i, (int)height % ngpu) - 1);
    // printf("%d %d %d\n",h / ngpu * i,std::min(i, h % ngpu),hbegin[i] );
    hend[i] = (int)height / ngpu * (i + 1) + std::min(i + 1, (int)height % ngpu) + 1;
    if (i == ngpu - 1)
      hend[i] = (int)height;
  }

  for (int i = 0; i < ngpu; i++)
  {
    CHECK_HIP(hipSetDevice(i));
    CHECK_HIP(hipMallocAsync(&sphere_gpu[i], spheres.size() * sizeof(SphereGPU), hipStreamDefault));

    CHECK_HIP(hipMemcpyAsync((void **)sphere_gpu[i], spheres.data(), spheres.size() * sizeof(Sphere), hipMemcpyHostToDevice));
  }
  // CHECK_HIP(hipHostMalloc(&sphere_gpu, spheres.size() * sizeof(SphereGPU), hipMemAllocationTypePinned));
  // CHECK_HIP(hipMemcpyAsync((void **)sphere_gpu, spheres.data(), spheres.size() * sizeof(Sphere), hipMemcpyHostToHost));

  CHECK_HIP(hipHostMalloc((void **)&image, width * height * sizeof(Vec3fGPU), hipMemAllocationTypePinned));
  float invWidth = 1 / float(width), invHeight = 1 / float(height);
  float fov = 30, aspectratio = width / float(height);
  float angle = tan(M_PI * 0.5 * fov / 180.);

  for (int i = 0; i < ngpu; i++)
  {
    CHECK_HIP(hipSetDevice(i));
    dim3 blockdim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 griddim((width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (hend[i] - hbegin[i] + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    raytrace_kernel<<<griddim, blockdim>>>(image, width, height, sphere_gpu[i], (int)spheres.size(), invWidth, invHeight, angle, aspectratio, hbegin[i], hend[i] - hbegin[i]);
  }
  // Trace rays
  for (int i = 0; i < ngpu; i++)
  {
    CHECK_HIP(hipSetDevice(i));
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipFree(sphere_gpu[i]));
  }

  return image;
}
void render_gpu_v2(Vec3fGPU *image[BUFFER_SIZE], const std::vector<Sphere> &spheres, size_t width, size_t height)
{
  // TODO:
  SphereGPU *sphere_gpu[2];

  // hipStream_t streams[BUFFER_SIZE];
  int ngpu;
  CHECK_HIP(hipGetDeviceCount(&ngpu));
  // printf("num GPUs: %d\n", ngpu);
  int hbegin[BUFFER_SIZE], hend[BUFFER_SIZE];
  for (int i = 0; i < BUFFER_SIZE; i++)
  {

    hbegin[i] = std::max(0, (int)height / BUFFER_SIZE * i + std::min(i, (int)height % BUFFER_SIZE) - 1);
    // printf("%d %d %d\n",h / ngpu * i,std::min(i, h % ngpu),hbegin[i] );
    hend[i] = (int)height / BUFFER_SIZE * (i + 1) + std::min(i + 1, (int)height % BUFFER_SIZE) + 1;
    if (i == BUFFER_SIZE - 1)
      hend[i] = (int)height;
  }

  for (int i = 0; i < ngpu; i++)
  {
    CHECK_HIP(hipSetDevice(i));
    CHECK_HIP(hipMallocAsync(&sphere_gpu[i], spheres.size() * sizeof(SphereGPU), hipStreamDefault));

    CHECK_HIP(hipMemcpyAsync((void **)sphere_gpu[i], spheres.data(), spheres.size() * sizeof(Sphere), hipMemcpyHostToDevice));
  }
  // CHECK_HIP(hipHostMalloc(&sphere_gpu, spheres.size() * sizeof(SphereGPU), hipMemAllocationTypePinned));
  // CHECK_HIP(hipMemcpyAsync((void **)sphere_gpu, spheres.data(), spheres.size() * sizeof(Sphere), hipMemcpyHostToHost));
  float invWidth = 1 / float(width), invHeight = 1 / float(height);
  float fov = 30, aspectratio = width / float(height);
  float angle = tan(M_PI * 0.5 * fov / 180.);

  for (int i = 0; i < BUFFER_SIZE; i++)
  {
    CHECK_HIP(hipHostMalloc((void **)&image[i], width * (hend[i] - hbegin[i]) * sizeof(Vec3fGPU), hipMemAllocationTypePinned));
    CHECK_HIP(hipSetDevice(i % ngpu));
    // CHECK_HIP(hipStreamCreate(&streams[i]));
    dim3 blockdim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 griddim((width + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (hend[i] - hbegin[i] + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);

    raytrace_kernel<<<griddim, blockdim, 0>>>(image[i], width, height, sphere_gpu[i % ngpu], (int)spheres.size(), invWidth, invHeight, angle, aspectratio, hbegin[i], hend[i] - hbegin[i]);
  }
  // Trace rays
  for (int i = 0; i < ngpu; i++)
  {
    CHECK_HIP(hipSetDevice(i));
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipFree(sphere_gpu[i]));
  }
}
void save_jpeg_image(const char *filename, Vec3fGPU *image, int image_width, int image_height);

void save_jpeg_image_v2(const char *filename, Vec3fGPU *image[], int image_width,
                        int image_height);
    // In the main function, we will create the scene which is composed of 5 spheres
    // and 1 light (which is also a sphere). Then, once the scene description is complete
    // we render that scene, by calling the render() function.
    int main(int argc, char **argv)
{
  size_t width;
  size_t height;
  char *filename = NULL;
  int verification = 0;

  struct timespec start, end, spent;
  clock_gettime(CLOCK_MONOTONIC, &start);

  if (argc < 3 || argc > 5)
  {
    fprintf(stderr, "$ ./raytracer <width> <height> <verification:(optional)0|1> <filename:(optional)>\n");
    return 1;
  }
  else
  {
    width = atoi(argv[1]);
    height = atoi(argv[2]);
    if (argc >= 3)
    {
      verification = atoi(argv[3]);
    }
    if (argc >= 4)
    {
      filename = argv[4];
    }
  }
  printf("image size width x height: %d x %d\n",width,height);

  std::vector<Sphere> spheres;
  // position, radius, surface color, reflectivity, transparency, emission color
  spheres.push_back(Sphere(Vec3f(0.0, -10004, -20), 10000, Vec3f(0.20, 0.20, 0.20), 0, 0.0));
  spheres.push_back(Sphere(Vec3f(0.0, 0, -20), 4, Vec3f(1.00, 0.32, 0.36), 1, 0.5));
  spheres.push_back(Sphere(Vec3f(5.0, -1, -15), 2, Vec3f(0.90, 0.76, 0.46), 1, 0.0));
  spheres.push_back(Sphere(Vec3f(5.0, 0, -25), 3, Vec3f(0.65, 0.77, 0.97), 1, 0.0));
  spheres.push_back(Sphere(Vec3f(-5.5, 0, -15), 3, Vec3f(0.90, 0.90, 0.90), 1, 0.0));
  // light
  spheres.push_back(Sphere(Vec3f(0.0, 20, -30), 3, Vec3f(0.00, 0.00, 0.00), 0, 0.0, Vec3f(3)));

  // Vec3fGPU *image_gpu = render_gpu(spheres, width, height);
  Vec3fGPU *image_gpu[BUFFER_SIZE];
  render_gpu_v2(image_gpu, spheres, width, height);
  // CHECK_HIP(hipHostMalloc((void **)&image_gpu, width * height * sizeof(Vec3fGPU), hipMemAllocationTypePinned));
  int hbegin[BUFFER_SIZE], hend[BUFFER_SIZE];
  for (int i = 0; i < BUFFER_SIZE; i++)
  {
    hbegin[i] = std::max(0, (int)height / BUFFER_SIZE * i + std::min(i, (int)height % BUFFER_SIZE) - 1);
    hend[i] = (int)height / BUFFER_SIZE * (i + 1) + std::min(i + 1, (int)height % BUFFER_SIZE) + 1;
    if (i == BUFFER_SIZE - 1)
      hend[i] = (int)height;
  }
  // clock_gettime(CLOCK_MONOTONIC, &end_gpu);
  float tolerance = 0.35;
  float diff = 0.0;
  size_t diff_cnt = 0;
  size_t total_cnt = width * height;
  if (verification > 0)
  {
    printf("\n=========Verification=========\n");
    Vec3f *image_gpu_for_eval;
    CHECK_HIP(hipHostMalloc(&image_gpu_for_eval, sizeof(Vec3f) * width * height, hipMemAllocationTypePinned));
    for (int b = 0; b < BUFFER_SIZE; ++b)
    {
      int begin = hbegin[b];
      int end = hend[b];
      if (begin == end)
        continue;
      CHECK_HIP(hipMemcpy(image_gpu_for_eval + begin * width, image_gpu[b],
                          sizeof(Vec3f) * width * (end - begin),
                          hipMemcpyHostToHost));
    }

    Vec3f *image_cpu = render_cpu(spheres, width, height);
    // #pragma omp parallel for num_threads(32)
    for (unsigned i = 0; i < width * height; ++i)
    {
      // total_cnt++;
      diff = abs(image_gpu_for_eval[i].x - image_cpu[i].x) + abs(image_gpu_for_eval[i].y - image_cpu[i].y) + abs(image_gpu_for_eval[i].z - image_cpu[i].z);
      if (diff > tolerance)
      {
        // printf("%d: diff(%f > %f), gpu(%f,%f,%f) cpu(%f,%f,%f)\n", i, diff, tolerance, image_gpu[i].x, image_gpu[i].y, image_gpu[i].z, image_cpu[i].x, image_cpu[i].y, image_cpu[i].z);
        diff_cnt++;
      }
    }

    double diff_rate = (double)diff_cnt / (double)total_cnt;
    printf("diff_cnt = %zu, correctness = %lf%%\n", diff_cnt, ((double)1.0 - diff_rate) * 100);
    if (diff_rate < 0.00001)
    {
      fprintf(stdout, "Verification Pass!\n");
    }
    else
    {
      fprintf(stdout, "Verification Failed\n");
    }
    printf("===============================\n\n");

    delete[] image_cpu;
  }

  if (filename != NULL)
  {
#if 0
    // Save result to a PPM image (keep these flags if you compile under Windows)
    std::ofstream ofs(filename, std::ios::out | std::ios::binary);
    ofs << "P6\n" << width << " " << height << "\n255\n";
    for (unsigned i = 0; i < width * height; ++i) {
      ofs << (unsigned char)(std::min(float(1), image_gpu[i].x) * 255) <<
        (unsigned char)(std::min(float(1), image_gpu[i].y) * 255) <<
        (unsigned char)(std::min(float(1), image_gpu[i].z) * 255);
    }
    ofs.close();
#else
    save_jpeg_image_v2(filename, image_gpu, width, height);
#endif
  }
  //   exit(0);
  // delete[] image_gpu;
  for (int i = 0; i < BUFFER_SIZE; i++)
    CHECK_HIP(hipFree(image_gpu[i]));
  clock_gettime(CLOCK_MONOTONIC, &end);
  // if (verification)
  // {
  //   timespec_subtract(&spent, &end, &end_gpu);
  //   printf("CPU Time: %ld.%09ld\n", spent.tv_sec, spent.tv_nsec);
  // }

  // timespec_subtract(&spent, &end_gpu, &start);
  // printf("GPU Time: %ld.%09ld\n", spent.tv_sec, spent.tv_nsec);

  timespec_subtract(&spent, &end, &start);

  printf("Elapsed Time: %ld.%09ld\n", spent.tv_sec, spent.tv_nsec);
  return 0;
}

typedef struct _RGB
{
  unsigned char r;
  unsigned char g;
  unsigned char b;
} RGB;

void save_jpeg_image(const char *filename, Vec3fGPU *image, int image_width, int image_height)
{
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  JSAMPROW row_pointer;

  RGB *rgb = (RGB *)malloc(sizeof(RGB) * image_width * image_height);
  for (int i = 0; i < image_width * image_height; ++i)
  {
    rgb[i].r = (unsigned char)(std::min((float)1, image[i].x) * 255);
    rgb[i].g = (unsigned char)(std::min((float)1, image[i].y) * 255);
    rgb[i].b = (unsigned char)(std::min((float)1, image[i].z) * 255);
  }

  int i;
  FILE *fp;

  cinfo.err = jpeg_std_error(&jerr);

  fp = fopen(filename, "wb");
  if (fp == NULL)
  {
    printf("Cannot open file to save jpeg image: %s\n", filename);
    exit(0);
  }

  jpeg_create_compress(&cinfo);

  jpeg_stdio_dest(&cinfo, fp);

  cinfo.image_width = image_width;
  cinfo.image_height = image_height;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;

  jpeg_set_defaults(&cinfo);

  jpeg_start_compress(&cinfo, TRUE);

  for (i = 0; i < image_height; i++)
  {
    row_pointer = (JSAMPROW)&rgb[i * image_width];
    jpeg_write_scanlines(&cinfo, &row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  fclose(fp);

  free(rgb);
}

void save_jpeg_image_v2(const char *filename, Vec3fGPU *image[], int image_width,
                        int image_height)
{
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  JSAMPROW row_pointer;

  RGB *rgb = (RGB *)malloc(sizeof(RGB) * image_width * image_height);

  int hbegin[BUFFER_SIZE], hend[BUFFER_SIZE];
  for (int i = 0; i < BUFFER_SIZE; i++)
  {
    hbegin[i] = std::max(0, (int)image_height / BUFFER_SIZE * i + std::min(i, (int)image_height % BUFFER_SIZE) - 1);
    hend[i] = (int)image_height / BUFFER_SIZE * (i + 1) + std::min(i + 1, (int)image_height % BUFFER_SIZE) + 1;
    if (i == BUFFER_SIZE - 1)
      hend[i] = (int)image_height;
  }

  for (int b = 0; b < BUFFER_SIZE; ++b)
  {
    int begin = hbegin[b];
    int end = hend[b];
    for (int x = begin; x < end; ++x)
    {
      for (int y = 0; y < image_width; ++y)
      {
        int gi = (x - begin) * image_width + y;
        int i = x * image_width + y;
        rgb[i].r = (unsigned char)(std::min((float)1, image[b][gi].x) * 255);
        rgb[i].g = (unsigned char)(std::min((float)1, image[b][gi].y) * 255);
        rgb[i].b = (unsigned char)(std::min((float)1, image[b][gi].z) * 255);
      }
    }
  }

  int i;
  FILE *fp;

  cinfo.err = jpeg_std_error(&jerr);

  fp = fopen(filename, "wb");
  if (fp == NULL)
  {
    printf("Cannot open file to save jpeg image: %s\n", filename);
    exit(0);
  }

  jpeg_create_compress(&cinfo);

  jpeg_stdio_dest(&cinfo, fp);

  cinfo.image_width = image_width;
  cinfo.image_height = image_height;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;

  jpeg_set_defaults(&cinfo);

  jpeg_start_compress(&cinfo, TRUE);

  for (i = 0; i < image_height; i++)
  {
    row_pointer = (JSAMPROW)&rgb[i * image_width];
    jpeg_write_scanlines(&cinfo, &row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  fclose(fp);

  free(rgb);
}