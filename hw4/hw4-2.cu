
//
//  hw4-2.cu
//  Includes gs2D-omp.cpp from hw2
//  Added the CUDA code to that
//
//
//  Created by Ilyeech Kishore Rapelli on 03/13/20.
//

#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <string>
#include <cooperative_groups.h>
using namespace cooperative_groups;

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}



#define BLOCK_SIZE 90

/*
__global__ void cg_jacobi_kernel(thread_group g, const double* f,double* u1_d,double* u2_d, double* diff_d, long N, long Loops){
  //__shared__ double smem[BLOCK_SIZE];
  double h = (double)1/(N-1);
  //int tid = threadIdx.x;
  int lane = g.thread_rank();
  int row = lane / N;
  int col = lane % N;
  for(int i=0; i < Loops; i++){
        if(i%2 == 0){
                if (row > 0 && row < N-1 && col > 0 && col < N-1)
                        u1_d[lane] = 0.25 * (h*h*f[lane] + u2_d[lane-1] + u2_d[lane+1] + u2_d[lane-N] + u2_d[lane+N]);
        } else {
                if (row > 0 && row < N-1 && col > 0 && col < N-1)
                        u2_d[lane] = 0.25 * (h*h*f[lane] + u1_d[lane-1] + u1_d[lane+1] + u1_d[lane-N] + u1_d[lane+N]);
        }
  g.sync();
  }
}


__global__ void cg_jacobi_kernel(const double* f,double* u1_d,double* u2_d, double* diff_d, long N, long Loops){
	//thread_block block = this_thread_block();
}

__global__ void naive_jacobi_kernel(const double* f,double* u1_d,double* u2_d, double* diff_d, long N, long Loops){
  //__shared__ double smem[BLOCK_SIZE];
  double h = (double)1/(N-1);
  //int tid = threadIdx.x;
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
  int row = idx / N;
  int col = idx % N;
  for(int i=0; i < Loops; i++){
	if(i%2 == 0){
		if (row > 0 && row < N-1 && col > 0 && col < N-1)
			u1_d[idx] = 0.25 * (h*h*f[idx] + u2_d[idx-1] + u2_d[idx+1] + u2_d[idx-N] + u2_d[idx+N]);
	} else {
		if (row > 0 && row < N-1 && col > 0 && col < N-1)
			u2_d[idx] = 0.25 * (h*h*f[idx] + u1_d[idx-1] + u1_d[idx+1] + u1_d[idx-N] + u1_d[idx+N]);
	}
  //if(threadIdx.x == 524) printf("hello from thread %d of block %d\n", threadIdx.x, blockIdx.x);
  	__syncthreads();
  }
}
*/

__global__ void onestep_jacobi_kernel(const double* f,double* u1_d,double* u2_d, double* diff_d, long N,long i, long Loops){
  //__shared__ double smem[BLOCK_SIZE];
  double h = (double)1/(N-1);
  //int tid = threadIdx.x;
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
  int row = idx / N;
  int col = idx % N;
	if(i%2 == 0){
		if (row > 0 && row < N-1 && col > 0 && col < N-1)
			u1_d[idx] = 0.25 * (h*h*f[idx] + u2_d[idx-1] + u2_d[idx+1] + u2_d[idx-N] + u2_d[idx+N]);
	} else {
		if (row > 0 && row < N-1 && col > 0 && col < N-1)
			u2_d[idx] = 0.25 * (h*h*f[idx] + u1_d[idx-1] + u1_d[idx+1] + u1_d[idx-N] + u1_d[idx+N]);
	}
}


int main(int argc, char** argv) {
    long N=1000;
    long Loops = 10000;
    printf("\nFor Jacobi calculation with N = %ld points\tLoops = %ld:\nCPU:\n", N, Loops);
    /*printf("Enter the no of points:");
    scanf("%ld", &N);
    printf("Enter maximum number of loops:");
    scanf("%ld", &Loops);*/
    double h = (double) 1/(N+1);
    //printf("\nN: %ld\th: %f\n", N, h);
    //double normResidual = 0;
    //double initialNormResidual = 0;
    
    double** f = (double **) malloc((N+2) * sizeof(double *));
    for(long j = 0; j <N+2; j++)
        f[j] = (double *) malloc((N+2) * sizeof(double));
    double** u1 = (double **) malloc((N+2) * sizeof(double *));
    for(long j = 0; j <N+2; j++)
        u1[j] = (double *) malloc((N+2) * sizeof(double));
    double** u2 = (double **) malloc((N+2) * sizeof(double *));
    for(long j = 0; j <N+2; j++)
        u2[j] = (double *) malloc((N+2) * sizeof(double));
    double** diff = (double **) malloc((N+2) * sizeof(double *));
    for(long j = 0; j <N+2; j++)
        diff[j] = (double *) malloc((N+2) * sizeof(double));
    double** temp;
    double** latest = u1;
    double** secondLatest = u2;

    long i = 0;
    double t;
    int nrT;
    double singleC = 1;
    int nrThreads[3] = {1, 4, 32};
    for(int nrTIndex = 0; nrTIndex < 3; nrTIndex++) {
	nrT = nrThreads[nrTIndex];
	//normResidual = 0;
	//initialNormResidual = 0;
	i = 0;
	latest = u1;
	secondLatest = u2;
	
	#ifdef _OPENMP
	t = omp_get_wtime();
	#endif
	
	for(long j = 0; j < N+2; j++){
	    for(long k =0; k <N+2; k++){
		f[j][k] = 1; //drand48()-0.5;//f is the vector of all 1's
		//printf("f[j][k]=%f\n", f[j][k]);
		u1[j][k] = 0;//do remember the boundary points are always zero.
		u2[j][k] = 0;
		diff[j][k] = 0;//It's unnecessary though
	    }
	}
	
	for(i=0; i < Loops; i++){
	    #pragma omp parallel for num_threads(nrT) schedule(guided, 4)
	    for(long j=1; j<N+1;j++){
		for(long k=1; k < N+1; k++){
			    secondLatest[j][k]= 0.25 * (h*h*f[j][k] + latest[j-1][k] + latest[j+1][k] + latest[j][k+1] + latest[j][k-1]);
		    //Jacobi algo for Laplace equations
		}
	    }
	    temp=latest;//switching pointers
	    latest = secondLatest;//latest points to the latest values now
	    secondLatest = temp;//second latest points to the second latest values
	   /* 
	    normResidual = 0;
	    #pragma omp parallel for num_threads(nrT) reduction(+:normResidual) schedule(guided, 4)
	    for(long j=1; j<N+1;j++){
		for(long k=1; k < N+1; k++){
		    diff[j][k] = (N+1)*(N+1)* (4*latest[j][k] - latest[j+1][k] - latest[j-1][k] - latest[j][k-1] - latest[j][k+1])
				- f[j][k];// a term in Au-f;
		    //printf("latest[j][k]: %f\t Norm Residual term: %f \t%ld\t%ld\n", latest[j][k], diff[j][k], j, k);
		    //printf("latest[%ld][%ld]: %f\t Norm Residual term: %10f\n", j, k, latest[j][k], diff[j][k]);

		    normResidual += diff[j][k]*diff[j][k];
		}
	    }
	    if(i==0){
		initialNormResidual = normResidual;
		//printf("Initial Norm Residual: %f\n", initialNormResidual);
	    } else {
	    //printf("%f\n", normResidual);
	    }
	    if(normResidual < initialNormResidual * 0.000001){
		printf("\nExiting the loop in iteration %ld \n", i);
		break;
	    }*/
	}

	//printf("Initial Norm Residual: \t%f \n", initialNormResidual);
	//printf("Ending Norm Residual: \t%f\n", normResidual);
	//printf("The loop ran %ld times.\n", i);
	
	#ifdef _OPENMP
	t = omp_get_wtime() - t;
	#endif
	
	if(nrT == 1)
	  singleC = t;
	printf("No of Threads %d: time elapsed = %f, speedup = %f\n", nrT, t, singleC/t);
	printf("No of Threads %d: CPU Bandwidth = %f GB/s\n", nrT, 2*N*N*i*sizeof(double) / 1e9 / t);
	printf("No of Threads %d: CPU Floprate = %f Gflops/s\n", nrT, 5*N*N*i / 1e9 / t);
    }
    
    u1 = latest;//To compare u1 with GPU computation
    u2 = secondLatest;
   

	double *f_1dim, *u1_1dim, *u2_1dim, *diff_1dim;
	f_1dim =  (double *) malloc((N+2) * (N+2) * sizeof(double *));
	u1_1dim =  (double *) malloc((N+2) * (N+2) * sizeof(double *));
	u2_1dim =  (double *) malloc((N+2) * (N+2) * sizeof(double *));
	diff_1dim =  (double *) malloc((N+2) * (N+2) * sizeof(double *));
 	
    //GPU version of Jacobi
    //__constant__ double *f_d;
    double *f_d;
    double *u1_d, *u2_d, *diff_d;
    cudaMalloc(&f_d, (N+2)*(N+2) * sizeof(double));
    Check_CUDA_Error("Malloc f failed");
    cudaMalloc(&u1_d, (N+2)*(N+2) * sizeof(double));
    cudaMalloc(&u2_d, (N+2)*(N+2) * sizeof(double));
    cudaMalloc(&diff_d, (N+2)*(N+2) * sizeof(double));
    //printf("u1_d, u2_d allocated\n");

	//dim2 GridDim(8,16);
	//int n1 = 8;
	//int n2 = 16;
	int n_blocks = ((N+2) * (N + 2) + BLOCK_SIZE - 1) / BLOCK_SIZE;
	
	t = omp_get_wtime();
	for(long j = 0; j < N+2; j++){
	    long j_inner = j * (N+2);
            for(long k =0; k <N+2; k++){
                f_1dim[j_inner + k] = 1; //drand48()-0.5;//f is the vector of all 1's
                u1_1dim[j_inner + k] = 0;//do remember the boundary points are always zero.
                u2_1dim[j_inner + k] = 0;
                diff_1dim[j_inner + k] = 0;//It's unnecessary though
            }
        }
	
	cudaMemcpy(f_d, f_1dim, (N+2)*(N+2) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(u1_d, u1_1dim, (N+2)*(N+2) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(u2_d, u2_1dim, (N+2)*(N+2) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(diff_d, diff_1dim, (N+2)*(N+2) * sizeof(double), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	//printf("Copied F, u1 to device\n");

	//Jacobi using Cooperative Groups
	//cg_jacobi_helper_kernel<<<n_blocks, BLOCK_SIZE>>>(f_d, u1_d, u2_d, diff_d, N+2, Loops);

	//dim2 BlockDim(n1, n2);
	//naive_jacobi_kernel<<<n_blocks, BLOCK_SIZE>>>(f_d, u1_d, u2_d, diff_d, N+2, Loops);
	//cudaDeviceSynchronize();
	//printf("\nGPU with a naive Jacobi Kernel:\n");
	
    for(long i = 0; i < Loops; i++){
	onestep_jacobi_kernel<<<n_blocks, BLOCK_SIZE>>>(f_d, u1_d, u2_d, diff_d, N+2, i, Loops);
	cudaDeviceSynchronize();
	//if(i % (Loops/10) == 0)
	//	printf("10 percent more done.\n");
    }
    
    
	cudaMemcpyAsync(u1_1dim, u1_d, (N+2)*(N+2)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(u2_1dim, u2_d, (N+2)*(N+2)*sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	#ifdef _OPENMP
	t = omp_get_wtime() - t;
	#endif

	printf("\nGPU:\n");
	int nDevices;
	cudaGetDeviceCount(&nDevices); // get number of GPUs
	for (int i = 0; i < nDevices; i++) { // loop over all GPUs
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i); // get GPU properties
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
	}
	printf("GPU Time Elapsed = %f\n", t);
	printf("GPU Bandwidth = %f GB/s\n", 2*N*N*Loops*sizeof(double) / 1e9 / t);
        printf("GPU Flop Rate: %f Gflops/s\n", 5*N*N*Loops / 1e9 / t);
	
	double* latest_d = u1_1dim;
 	double* secondLatest_d = u2_1dim;
	
	if(Loops % 2 == 0)
		latest_d = secondLatest_d;//The Final solution
	double err = 0;
	for (long i = 0; i < (N+2) * (N+2); i++) err += fabs(latest[i/(N+2)][i % (N+2)] - latest_d[i]);
	printf("Error = %f\n", fabs(err));
	for (long i = 0; i < (N+2) * (N+2); i++){
		err = fabs(latest[i/(N+2)][i % (N+2)] - latest_d[i]);
		if(err > 0.00001)
			printf("latest[%ld][%ld] = %f, latest_d[%ld][%ld] = %f\n",i/(N+2), i%(N+2), latest[i/(N+2)][i % (N+2)], i/(N+2), i%(N+2), latest_d[i]);
	}
    //Deallcoation
    for(long j = 0; j <N+2; j++){
	free(f[j]);
	free(u1[j]);
	free(u2[j]);
	free(diff[j]);
    }
    free(f);
    free(u1);
    free(u2);
    free(diff);

	free(f_1dim);
	free(u1_1dim);
	free(u2_1dim);
	free(diff_1dim);

	checkCuda( cudaFree(f_d));	
	checkCuda( cudaFree(u1_d));	
	checkCuda( cudaFree(u2_d));	
	checkCuda( cudaFree(diff_d));	
}
