#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>

void glaxpy(int n, float a, float *x, float *y)
{
  for (int i = 0; i < n; ++i)
      y[i] = a*x[i] + y[i];
}


__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  cudaMalloc(&d_x, N*sizeof(float));
  cudaMalloc(&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaEvent_t start, stop;



  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

  cudaEventRecord(start);
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
  cudaEventRecord(stop);

  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);
  printf("Time: %f\n", milliseconds);
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);


// lets do cpu
  int NN = 1<<20;
  float *xx, *yy;
  xx = (float*)malloc(NN*sizeof(float));
  yy = (float*)malloc(NN*sizeof(float));

  for (int i = 0; i < NN; i++) {
    xx[i] = 1.0f;
    yy[i] = 2.0f;
  }


  cudaEvent_t sttart, sttop;
  cudaEventCreate(&sttart);
  cudaEventCreate(&sttop);
  cudaEventRecord(sttart);
  glaxpy(N, 2.0f, xx, yy);
  cudaEventRecord(sttop);

  // cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

  cudaEventSynchronize(sttop);
  float miilliseconds = 0;
  cudaEventElapsedTime(&miilliseconds, sttart, sttop);

    // float maxError = 0.0f;
    // for (int i = 0; i < N; i++)
    //   maxError = max(maxError, abs(y[i]-4.0f));
    // printf("Max error: %f\n", maxError);
  printf("Timecpp: %f\n", miilliseconds);
    // cudaFree(d_x);
    // cudaFree(d_y);
  free(xx);
  free(yy);
}
