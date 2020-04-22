// $ nvcc -arch=sm_30 hw4-1a.cu -o hw4-1a -Xcompiler -fopenmp
// flag -Xcompiler passes next flag directly to compiler
#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

void vec_dot_product(double* c, const double* a, const double* b, long N){
  #pragma omp parallel for schedule(static) reduction(+:c[0])
  for (long i = 0; i < N; i++) {
    c[0] += a[i] * b[i];
  }
}

#define BLOCK_SIZE 1024
__device__ void warpReduce(volatile double* smem, int tid) {
	//Loop unrolling
	//printf("Warp Reduce at tid: %d\n", tid);
	smem[tid] += smem[tid + 32];
	smem[tid] += smem[tid + 16];
	smem[tid] += smem[tid + 8];
	smem[tid] += smem[tid + 4];
	smem[tid] += smem[tid + 2];
	smem[tid] += smem[tid + 1];
  }


__global__ void vec_dot_product_kernel(double* c, const double* a, const double* b, long N){
  __shared__ double smem[BLOCK_SIZE];
  int tid = threadIdx.x;
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < N){
	//printf("Block %d: smem[%d] = a[%d] * b[%d] == %f += %f * %f\n", blockIdx.x, tid, idx, idx, smem[tid], a[idx], b[idx]);
	smem[tid] = a[idx] * b[idx];
	//printf("Block %d: smem[%d] = %f\n", blockIdx.x, tid, smem[tid]);
  } else	smem[tid] = 0;
  //if(threadIdx.x == 524) printf("hello from thread %d of block %d\n", threadIdx.x, blockIdx.x);
  __syncthreads();
  for(unsigned int s = blockDim.x/2; s>32; s>>=1){
	if(tid < s) {
		smem[tid] += smem[tid+s];
	}
	__syncthreads();
  }
  if(tid < 32) warpReduce(smem, tid);
  //if(tid <1) smem[0] = smem[0] + smem[1];
  if(tid == 0){
	c[blockIdx.x] = smem[tid];
	//printf("Block %d: Updated c[%d] to: %f\n", blockIdx.x, blockIdx.x, smem[tid]);
  }
}

__global__ void reduction_kernel0(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  int tid = threadIdx.x;
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
  // if(threadIdx.x == 524) printf("hello from thread %d of block %d\n", threadIdx.x, blockIdx.x);
  if (idx < N) smem[tid] = a[idx];
  else smem[tid] = 0;
  __syncthreads();
  for(int s = blockDim.x/2; s > 0; s >>=1) {
	  if(tid < s)
		  smem[tid] += smem[tid + s];
	  __syncthreads();
  }
  if (tid == 0){
	sum[blockIdx.x] = smem[tid];
	//printf("Block %d: Updated sum[%d] to: %f\n", blockIdx.x, blockIdx.x, smem[tid]);
  }
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

int main() {

  long N = (1UL<<25);//64*64*BLOCK_SIZE;//(1UL<<27); // 2^25
  printf("\nFor Vector Vector dot product with N = %ld:\n", N);
  double* x = (double*) malloc(N * sizeof(double));
  double* y = (double*) malloc(N * sizeof(double));
  double* z = (double*) malloc(1 * sizeof(double));
  //double* z = (double*) malloc(N * sizeof(double));
  double* z_ref = (double*) malloc(1 * sizeof(double));
  //double* z_ref = (double*) malloc(N * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    x[i] = i+2;
    y[i] = 1/(i+1);
    //z[i] = 0;
    //z_ref[i] = 0;
  }
  z[0] = 0;
  z_ref[0] = 0;

  double tt = omp_get_wtime();
  vec_dot_product(z_ref, x, y, N);
  printf("CPU Bandwidth = %f GB/s\n", 2*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  //printf("Dot product from CPU: %f\n", z_ref[0]);


  double *x_d, *y_d, *z_d, *dot_d;
  cudaMalloc(&x_d, N*sizeof(double));
  Check_CUDA_Error("malloc x failed");
  cudaMalloc(&y_d, N*sizeof(double));
  cudaMalloc(&z_d, 1*sizeof(double));
  long N_work = 1;
  for (long i = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += i;
  cudaMalloc(&dot_d, N_work*sizeof(double)); // extra memory buffer for reduction across thread-blocks

  tt = omp_get_wtime();
  cudaMemcpy(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
  //Following page 18 of https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
  cudaDeviceSynchronize();
  double* dot_product_d = dot_d;
  long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
  //Dot Product + A bit of reduction
  vec_dot_product_kernel<<<Nb,BLOCK_SIZE>>>(dot_product_d, x_d, y_d, N);
  
  //It's just reduction now onwards.
  while (Nb > 1) {
    long N = Nb;
    Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
    reduction_kernel0<<<Nb,BLOCK_SIZE>>>(dot_product_d + N, dot_product_d, N);
    dot_product_d += N;
  }

  cudaMemcpyAsync(z, dot_product_d, 1*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("GPU Bandwidth = %f GB/s\n", 2*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  //printf("GPU Flop Rate = %f GFlops/s\n", N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  //printf("Dot product from GPU: %f\n", z[0]);
  double err = 0;
  //for (long i = 0; i < N; i++) err += fabs(z[i]-z_ref[i]);
  err = z[0] - z_ref[0];
  printf("Error = %f\n", fabs(err));
  //printf("GPU: %f\n", z[0]);

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(dot_d);
  cudaFree(z_d);

  free(x);
  free(y);
  free(z);
  free(z_ref);
  return 0;
}
