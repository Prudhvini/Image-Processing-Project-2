// Forward and inverse FFT code
// Hacked for CImg by D. Crandall
// 

// Compute 2D Discrete Fourier Transform (DFT) using the Fast Fourier
//  Transform (FFT) algorithm.
// dir: 1 for forward transform, 0 for inverse
// real and imag buffers are *overwritten*
//
#include "CImg.h"
using namespace cimg_library;
using namespace std;
void FFT_1D(short int dir,long m,double *x,double *y);

void swap_quadrants(CImg<double> &real, CImg<double> &imag)
{
  for(int i=0; i<real.height()/2; i++)
    {
      for(int j=0; j<real.width()/2; j++)
	{
	  swap(real(j,i), real(j+real.width()/2, i+real.height()/2));
	  swap(imag(j,i), imag(j+imag.width()/2, i+imag.height()/2));
	}
      for(int j=real.width()/2; j<real.width(); j++)
	{
	  swap(real(j,i), real(j-real.width()/2, i+real.height()/2));
	  swap(imag(j,i), imag(j-imag.width()/2, i+imag.height()/2));
	}
    }
}

void FFT_2D(short int dir, CImg<double> &real, CImg<double> &imag)
{
  // check squareness and powers of two
  if(real.height() != real.width() || !real.is_sameXYZC(imag) ||
     (real.height() & (real.height()-1)) || real.spectrum() != 1)
    throw string("In fft, image isn't square and/or size isn't a power of 2!");

  // figure exponent k
  int k = int(round(log2(real.height())));

  if(!dir)
    swap_quadrants(real, imag);

  // 1d transform on rows
  for(int j=0; j<real.height(); j++)
    FFT_1D(dir, k, real.data(0, j, 0, 0), imag.data(0, j, 0, 0));

  // 1d transform on cols
  real.transpose(); imag.transpose();
  for(int j=0; j<real.height(); j++)
    FFT_1D(dir, k, real.data(0, j, 0, 0), imag.data(0, j, 0, 0));

  real.transpose(); imag.transpose();

  if(dir)
    swap_quadrants(real, imag);
}


// This function is from
// http://paulbourke.net/miscellaneous/dft/
//
void FFT_1D(short int dir,long m,double *x,double *y)
{
  long n,i,i1,j,k,i2,l,l1,l2;
  double c1,c2,tx,ty,t1,t2,u1,u2,z;

  /* Calculate the number of points */
  n = 1;
  for (i=0;i<m;i++) 
    n *= 2;

  /* Do the bit reversal */
  i2 = n >> 1;
  j = 0;
  for (i=0;i<n-1;i++) {
    if (i < j) {
      tx = x[i];
      ty = y[i];
      x[i] = x[j];
      y[i] = y[j];
      x[j] = tx;
      y[j] = ty;
    }
    k = i2;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j += k;
  }
  
  /* Compute the FFT */
  c1 = -1.0; 
  c2 = 0.0;
  l2 = 1;
  for (l=0;l<m;l++) {
    l1 = l2;
    l2 <<= 1;
    u1 = 1.0; 
    u2 = 0.0;
    for (j=0;j<l1;j++) {
      for (i=j;i<n;i+=l2) {
	i1 = i + l1;
	t1 = u1 * x[i1] - u2 * y[i1];
	t2 = u1 * y[i1] + u2 * x[i1];
	x[i1] = x[i] - t1; 
	y[i1] = y[i] - t2;
	x[i] += t1;
	y[i] += t2;
      }
      z =  u1 * c1 - u2 * c2;
      u2 = u1 * c2 + u2 * c1;
      u1 = z;
    }
    c2 = sqrt((1.0 - c1) / 2.0);
    if (dir == 1) 
      c2 = -c2;
    c1 = sqrt((1.0 + c1) / 2.0);
  }
  
  /* Scaling for forward transform */
  if (dir == 1) {
    for (i=0;i<n;i++) {
      x[i] /= n;
      y[i] /= n;
    }
  }

}


