#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timers.h"

#define COUNT_MAX   2000
#define MIN(x,y)    ((x) < (y) ? (x) : (y))

#ifdef SAVE_JPG
void save_jpeg_image(const char* filename, int* r, int* g, int* b, int image_width, int image_height);
#endif

typedef struct {
  unsigned int r;
  unsigned int g;
  unsigned int b;
} RgbColor;

RgbColor HSVtoRGB(unsigned h, unsigned s, unsigned v);


// Main part of the below code is originated from Lode Vandevenne's code.
// Please refer to http://lodev.org/cgtutor/juliamandelbrot.html
void julia(int w, int h, char* output_filename) {
  // each iteration, it calculates: new = old*old + c,
  // where c is a constant and old starts at current pixel

  // real and imaginary part of the constant c
  // determinate shape of the Julia Set
  double cRe, cIm;

  // real and imaginary parts of new and old
  double newRe, newIm, oldRe, oldIm;

  // you can change these to zoom and change position
  double zoom = 1, moveX = 0, moveY = 0;

  // after how much iterations the function should stop
  int maxIterations = COUNT_MAX;

#ifndef SAVE_JPG
  FILE *output_unit;
#endif

  double wtime;

  // pick some values for the constant c
  // this determines the shape of the Julia Set
  cRe = -0.7;
  cIm = 0.27015;

  int *r = (int *)calloc(w * h, sizeof(int));
  int *g = (int *)calloc(w * h, sizeof(int));
  int *b = (int *)calloc(w * h, sizeof(int));

  printf( "  Sequential C version\n" );
  printf( "\n" );
  printf( "  Create an ASCII PPM image of the Julia set.\n" );
  printf( "\n" );
  printf( "  An image of the set is created using\n" );
  printf( "    W = %d pixels in the X direction and\n", w );
  printf( "    H = %d pixels in the Y direction.\n", h );

  timer_init();
  timer_start(0);

  // loop through every pixel
  for (int y = 0; y < h; y++)
  {
    for (int x = 0; x < w; x++)
    {
      // calculate the initial real and imaginary part of z,
      // based on the pixel location and zoom and position values
      newRe = 1.5 * (x - w / 2) / (0.5 * zoom * w) + moveX;
      newIm = (y - h / 2) / (0.5 * zoom * h) + moveY;

      // i will represent the number of iterations
      int i;
      //start the iteration process
      for (i = 0; i < maxIterations; i++)
      {
        // remember value of previous iteration
        oldRe = newRe;
        oldIm = newIm;

        // the actual iteration, the real and imaginary part are calculated
        newRe = oldRe * oldRe - oldIm * oldIm + cRe;
        newIm = 2 * oldRe * oldIm + cIm;

        // if the point is outside the circle with radius 2: stop
        if ((newRe * newRe + newIm * newIm) > 4) break;
      }

      // use color model conversion to get rainbow palette, 
      // make brightness black if maxIterations reached
      RgbColor color = HSVtoRGB(i % 256, 255, 255 * (i < maxIterations));
      r[y*w+x] = color.r;
      g[y*w+x] = color.g;
      b[y*w+x] = color.b;     
    }
  }

  timer_stop(0);
  wtime = timer_read(0);
  printf( "\n" );
  printf( "  Time = %lf seconds.\n", wtime );

#ifdef SAVE_JPG
  save_jpeg_image(output_filename, r, g, b, w, h);
#else
  // Write data to an ASCII PPM file.
  output_unit = fopen( output_filename, "wt" );

  fprintf( output_unit, "P3\n" );
  fprintf( output_unit, "%d  %d\n", h, w );
  fprintf( output_unit, "%d\n", 255 );
  for ( int i = 0; i < h; i++ )
  {
    for ( int jlo = 0; jlo < w; jlo = jlo + 4 )
    {
      int jhi = MIN( jlo + 4, w );
      for ( int j = jlo; j < jhi; j++ )
      {
        fprintf( output_unit, "  %d  %d  %d", r[i*w+j], g[i*w+j], b[i*w+j] );
      }
      fprintf( output_unit, "\n" );
    }
  }

  fclose( output_unit );
#endif
  printf( "\n" );
  printf( "  Graphics data written to \"%s\".\n\n", output_filename );

  // Terminate.
  free(r);
  free(g);
  free(b);
}


RgbColor HSVtoRGB(unsigned h, unsigned s, unsigned v)
{
  RgbColor rgb;
  unsigned char region, remainder, p, q, t;

  if (s == 0)
  {
    rgb.r = v;
    rgb.g = v;
    rgb.b = v;
    return rgb;
  }

  region = h / 43;
  remainder = (h - (region * 43)) * 6; 

  p = (v * (255 - s)) >> 8;
  q = (v * (255 - ((s * remainder) >> 8))) >> 8;
  t = (v * (255 - ((s * (255 - remainder)) >> 8))) >> 8;

  switch (region)
  {
    case 0:
      rgb.r = v; rgb.g = t; rgb.b = p;
      break;
    case 1:
      rgb.r = q; rgb.g = v; rgb.b = p;
      break;
    case 2:
      rgb.r = p; rgb.g = v; rgb.b = t;
      break;
    case 3:
      rgb.r = p; rgb.g = q; rgb.b = v;
      break;
    case 4:
      rgb.r = t; rgb.g = p; rgb.b = v;
      break;
    default:
      rgb.r = v; rgb.g = p; rgb.b = q;
      break;
  }

  return rgb;
}

