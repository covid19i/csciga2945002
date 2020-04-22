// $ nvcc -arch=sm_30 hw4-1b.cu -o hw4-1b -Xcompiler -fopenmp
// flag -Xcompiler passes next flag directly to compiler
#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
//#include <cooperative_groups.h>
//using namespace cooperative_groups;

#define BLOCK_SIZE 1024

void vec_mat_mult(double* c, const double* a, const double* b, long N, long M){
  for(long j = 0; j < M; j++) {
  	#pragma omp parallel for schedule(static) reduction(+:c[j])
  	for (long i = 0; i < N; i++) {
    		c[j] += a[j*M + i] * b[i];
  	}
  }
}

//THIS IS THE ONE USED. NOT THE ONE ABOVE
void vec_mat_mult2dim(double* c, const double* a, const double* b, long N, long M){
  #pragma omp parallel for reduction(+:c[0:M])
  for(long j = 0; j < M; j++) {
  	for (long i = 0; i < N; i++) {
    		c[j] += a[j*N + i] * b[i];
  	}
  }
}

__global__ void vec_mat_mult_kernel(double* c, const double* a, const double* b, long N, long M){
	//THIS HAS A RACE CONDITION when updating c[idx/N]
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(threadIdx.x == 1) printf("hello from thread %d of block %d\n", threadIdx.x, blockIdx.x);
  if (idx < M*N) {
	//printf("c[%ld] += a[%d] * b[%ld] == %f += %f * %f\n", idx/N, idx, idx%N, c[idx/N], a[idx], b[idx%N]);
	c[idx/N] += a[idx] * b[idx%N];
  }
}

__device__ void warpReduce(volatile double* smem, int tid){
	//Loop unrolling
	smem[tid] += smem[tid + 32];
	smem[tid] += smem[tid + 16];
	smem[tid] += smem[tid + 8];
	smem[tid] += smem[tid + 4];
	smem[tid] += smem[tid + 2];
	smem[tid] += smem[tid + 1];

}
__global__ void mat_product_kernel(double* c, const double* a, const double* b, long N, long M){
  __shared__ double smem[BLOCK_SIZE];//WILL THIS CAUSE INITIALIZATION ERROR? UNDEFINED.
  int tid = threadIdx.x;
  smem[tid] = 0;
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
	int width_of_row = (((int)((N + BLOCK_SIZE - 1)/BLOCK_SIZE))*BLOCK_SIZE);
	int m = (idx / width_of_row);//Change to LONG if needed??????
	int n = (idx % width_of_row);
  if (n < N){
	smem[tid] = a[m * N + n] * b[n];
	//printf("Block %d: smem[%d] = %f\n", blockIdx.x, tid, smem[tid]);
  } else	smem[tid] = 0;
  //if(threadIdx.x == 524) printf("hello from thread %d of block %d\n", threadIdx.x, blockIdx.x);
  __syncthreads();
  for(unsigned int s = blockDim.x/2; s>0; s>>=1){
	if(tid < s) {
		smem[tid] += smem[tid+s];
	}
	__syncthreads();
  }
  //if(tid <32) warpReduce(smem, tid);
  if(tid == 0){
	c[blockIdx.x] = smem[tid];
        //printf("Block %d: Updated c[%d] to: %f\n", blockIdx.x, blockIdx.x, smem[tid]);
	//if(blockIdx.x == 0)
	//	printf("For above calculations: width_of_row = %d\tN = %ld\tM = %ld\n", width_of_row, N, M);
  }
}

__global__ void reduction_kernel1(double* sum, const double* a, long N, long M){
  __shared__ double smem[BLOCK_SIZE];
  int tid = threadIdx.x;
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
	int width_of_row = (((int)((N + BLOCK_SIZE - 1)/BLOCK_SIZE))*BLOCK_SIZE);
	int m = (idx / width_of_row);//Change to LONG if needed??????
	int n = (idx % width_of_row);
  if (n < N){
	smem[tid] = a[m * N + n];
	//printf("Reduction kernel: Block %d: smem[%d] = %f\n", blockIdx.x, tid, smem[tid]);
  } else	smem[tid] = 0;
  __syncthreads();
  for(int s = blockDim.x/2; s > 0; s >>=1) {
          if(tid < s)
                  smem[tid] += smem[tid + s];
          __syncthreads();
  }
  if (tid == 0){
        sum[blockIdx.x] = smem[tid];
        //printf("Block %d: Updated sum[%d] to: %f\n", blockIdx.x, blockIdx.x, smem[tid]);
	//if(blockIdx.x == 0)
	//	printf("For above calculations: width_of_row = %d\tN = %ld\tM = %ld\n", width_of_row, N, M);
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

  long N = (1UL<<25);//323*BLOCK_SIZE;//(1UL<<17); // 2^25
  long M = 17;
  printf("\nFor Matrix Vector multiplication with N = %ld\t M = %ld\n", N, M);

  double* x = (double*) malloc(M * N * sizeof(double));
  double* y = (double*) malloc(N * sizeof(double));
  double* z = (double*) malloc(M * sizeof(double));
  double* z_ref = (double*) malloc(M * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    for (long j = 0; j < M; j++) {
	x[i+j*N] = i+2;
    }
    y[i] = 1.0/(i+1);
  }
  for (long j = 0; j < M; j++) {
        z[j] = 0;
        z_ref[j] = 0;
  }

  double tt = omp_get_wtime();
  vec_mat_mult2dim(z_ref, x, y, N, M);
  printf("CPU Bandwidth = %f GB/s\n", (M*N + M + N) *sizeof(double) / (omp_get_wtime()-tt)/1e9);
  //printf("Mat Mult[0] from CPU: %f\n", z_ref[0]);
  //printf("Mat Mult[%ld] from CPU: %f\n", M-1, z_ref[M-1]);


  double *x_d, *y_d, *mat_d;
  cudaMalloc(&x_d, M*N*sizeof(double));
  Check_CUDA_Error("malloc x failed");
  cudaMalloc(&y_d, N*sizeof(double));
  long N_work = 1;
  for (long i = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += i;
	N_work *= M;
	//printf("Size of buffer memory: %ld doubles\n", N_work);
  cudaMalloc(&mat_d, M * N_work*sizeof(double)); // extra memory buffer for reduction across thread-blocks

  tt = omp_get_wtime();
  cudaMemcpy(x_d, x, M*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  double* mat_product_d = mat_d;
  long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
  //Dot Product + A bit of reduction
  //Will be Following page 18 of https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
  mat_product_kernel<<<(N*M+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(mat_product_d, x_d, y_d, N, M);
  cudaDeviceSynchronize();
  
  //It's just reduction now onwards.
  while (Nb > 1) {
    long N = Nb;
    Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
//	printf("Size of buffer memory: %ld doubles\n", Nb * M);
    reduction_kernel1<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE * M,BLOCK_SIZE>>>(mat_product_d + N*M, mat_product_d, N, M);
    mat_product_d += N * M;
  }

  cudaMemcpyAsync(z, mat_product_d, M*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("GPU Bandwidth = %f GB/s\n", (M*N + M + N) *sizeof(double) / (omp_get_wtime()-tt)/1e9);
  //printf("Mat Mult[0] from GPU: %f\n", z[0]);
  //printf("Mat Mult[%ld] from GPU: %f\n", M-1, z[M-1]);
  double err = 0;
  for (long i = 0; i < M; i++) err += fabs(z[i]-z_ref[i]);
  printf("Error = %f\n", fabs(err));

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(mat_d);
  //cudaFree(z_d);

  free(x);
  free(y);
  free(z);
  free(z_ref);
  return 0;
}
