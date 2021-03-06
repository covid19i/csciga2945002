//
//

#include "dataReader.h"
#include "PSGD.h"
#include "MultiLog.h"
#include "LossType.h"

#include <iostream>
#include <cstdlib>
#include <string>
#include <random>
#include <chrono>
#include <omp.h>
#include <vector>
#include <curand_kernel.h>
#include <curand.h>


using namespace std;
typedef unsigned char uchar;


double getLoss(double* weight,double** trainingData,uchar* trainingLabel,int n_data,int n_weights,int n_labels,double lambda){

    double summ=0;
    double* exponent=(double*)malloc(n_labels*sizeof(double));
    for(int i=0;i<n_data;i++){
        
        double expSum=0;
        for(int k=0;k<n_labels;k++){
            exponent[k]=0;
            for(int j=0;j<n_weights;j++){
                exponent[k]+=weight[k*n_weights+j]*trainingData[i][j];
                if(k==trainingLabel[i]){
                    summ-=exponent[k];
                }
            }
            expSum+=exp(exponent[k]);
        }
        if(expSum>0)summ+=log(expSum);
    }
    
    double regularizationTerm=0;
    for(int k=0; k<n_labels; k++){
        for(int j=0; j<n_weights; j++){
            regularizationTerm += weight[k*n_weights+j] * weight[k*n_weights+j];
        }
    }
    
    regularizationTerm*=lambda;
    return (regularizationTerm+summ)/n_data;
}

__device__ void warpReduce(volatile double* smem, int tid) {
  smem[tid] += smem[tid + 32];
  smem[tid] += smem[tid + 16];
  smem[tid] += smem[tid + 8];
  smem[tid] += smem[tid + 4];
  smem[tid] += smem[tid + 2];
  smem[tid] += smem[tid + 1];
}
#define BLOCK_SIZE 1024
#define QUERY 432 //make sure this is less than BLOCK_SIZE
__global__  void run_hogwild_one_processor(double* weight, double* trainingData, uchar* trainingLabel, double eta, int n_data, const int n_weights, const int n_labels,const double lambda,const long loop, const int total_loops) {
  __shared__ double smem[BLOCK_SIZE];//Gonna update 100% of weights in this kernel though.
  __shared__ double denominator;
  __shared__ double numerator[10];
  int tid = threadIdx.x;
  //printf("tid = %d\n", tid);
  //printf("Block %d: smem[%d] = a[%d] * b[%d] == %f += %f * %f\n", blockIdx.x, tid, idx, idx, smem[tid], a[idx], b[idx]);
  curandState_t state;
  curand_init(loop*loop, tid+blockIdx.x*blockDim.x, 0, &state);
  //curand_init(loop, tid, 0, &state);
  __shared__ int r;
  if(tid == 0){
    //printf("Loop: %ld\n", loop);
    r = curand(&state) % n_data;
    if(loop == 0) r = 24356;
    //printf("__shared__ r = %d for thread id: %d\n", r, tid);
  }
  for(int i=0; i < n_labels; i++){
    if(tid < n_weights){
      smem[tid] = weight[i*n_weights + tid] * trainingData[r * n_weights + tid];
    } else {
      smem[tid] = 0;
    }
    //printf("Block %d: smem[%d] = %f\n", blockIdx.x, tid, smem[tid]);
    __syncthreads();
    for(unsigned int s = blockDim.x/2; s>32; s>>=1){
        if(tid < s) {
          smem[tid] += smem[tid+s];
        }
        __syncthreads();
    }
    if(tid < 32) warpReduce(smem, tid);//Minimum BLOCK_SIZE = 64 because of this. BLOCK_SIZE has to be a power of 2
    __syncthreads();
    if(tid == 0){
      numerator[i] = smem[tid];
    }
    __syncthreads();
  }
  if(tid < n_labels)
    numerator[tid] = exp(numerator[tid]);
  __syncthreads();
  if(tid == 0){
    denominator = numerator[0] + numerator[1] + numerator[2] + numerator[3] + numerator[4] +
			numerator[5] + numerator[6] + numerator[7] + numerator[8] + numerator[9];
    //printf("Denominator = %f\n", denominator);
  }
  __syncthreads();
  if(tid < n_labels){
    numerator[tid] = ((trainingLabel[r] == tid)?1:0) - numerator[tid] / denominator;//Not just the numerator anymore
  }
  __syncthreads();
    for(int j=0; j < n_labels; j++){
      if(tid < 785){//BLOCK_SIZE can't be 785
        //Lock free
        /*if(tid == QUERY && j == 0 && (loop < 3 || loop % (total_loops/5) == 0 || loop == (total_loops-1) ) ){
          printf("first term for weight[%d]= %.10f\t for label %d\t block: %d\n", tid, -(numerator[j]) * trainingData[r * n_weights + tid], j, blockIdx.x);
          printf("second term for weight[%d]= %.20f\t for label %d\t block: %d\n", tid, lambda * 2 * weight[j * n_weights + tid] / n_data, j, blockIdx.x);
        }*/
        weight[j * n_weights + tid] -= eta * ( -(numerator[j]) * trainingData[r * n_weights + tid] +
                                             (lambda * 2 * weight[j * n_weights + tid] / n_data) );//1/n_data makes a difference?
      }
    }
  __syncthreads();

}

