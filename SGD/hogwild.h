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
    return regularizationTerm+summ;
}

#define BLOCK_SIZE 1024

__device__ void warpReduce(volatile double* smem, int tid) {
  smem[tid] += smem[tid + 32];
  smem[tid] += smem[tid + 16];
  smem[tid] += smem[tid + 8];
  smem[tid] += smem[tid + 4];
  smem[tid] += smem[tid + 2];
  smem[tid] += smem[tid + 1];
}

__global__  void run_hogwild_one_processor(double* weight, const double* trainingData, const double* trainingLabel, int data_point, double eta, int n_data, int n_weights, int n_labels, double lambda) {
  __shared__ double smem[n_weights];//Gonna update 100% of weights in this kernel though.
  __shared__ double denominator = 0;
  __shared__ double numerator[n_labels];
  __shared__ double indicator[n_labels];
  int tid = threadIdx.x;
  //printf("Block %d: smem[%d] = a[%d] * b[%d] == %f += %f * %f\n", blockIdx.x, tid, idx, idx, smem[tid], a[idx], b[idx]);
  for(int i=0; i < n_labels; i++){
    if(tid < n_weights){
      smem[tid] = weight[i*n_weights + tid] * trainingData[data_point * n_weights + tid];
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
    if(tid < 32) warpReduce(smem, tid);
    __syncthreads();
    if(tid == 0){
      numerator[i] = smem[tid];
    }
    __syncthreads();
  }
  if(tid < n_labels)
    numerator[tid] = exp(numerator[tid]);
  __syncthreads();
  if(tid == 0)
    denominator = numerator[0] + numerator[1] + numerator[2] + numerator[3] + numerator[4] +
			numerator[5] + numerator[6] + numerator[7] + numerator[8] + numerator[9];
  __syncthreads();
  if(tid < n_labels){
    numerator[tid] = numerator[tid] / denominator;
    indicator[tid] = ((trainingLabel[data_point] == tid)?1:0);
  __syncthreads();
    for(int j=0; j < n_labels; j++){
      //Lock free
      weight[j * n_weights + tid] -= eta * ( (indicator[j] - numerator[j]) * trainingData[data_point * n_weights + tid] +
                                             (lambda * 2 * weight[j * n_weights + tid] / n_data) );//1/n_data makes a difference?
    }
  }
  __syncthreads();

}


__global__ void updateWeightKernel(double* weight,const double* trainingData,const uchar* trainingLabel,double eta,int n_data,int n_weights,int n_labels,int batchSize,double lambda,int offset){
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    int index=(x*gridDim.x*blockDim.x+y)+offset;
    int weight_size=n_weights*n_labels;
    if(index<weight_size){
        double deltaWeight;
        double* data;
        data=(double*)malloc(batchSize*n_weights*sizeof(double));
        
        uchar* label;
        label=(uchar*)malloc(batchSize*sizeof(uchar));
        
        for(int b=0;b<batchSize;b++){
            curandState_t state;
            curand_init(index,0,b,&state);
            int r;
            r=curand(&state)%n_data;
            label[b]=trainingLabel[r];
            for(int w=0;w<n_weights;w++){
                data[b*n_weights+w]=trainingData[r*n_weights+w];
            }
        }
        deltaWeight=getOneGradient(weight,index, data, label,eta, batchSize, n_weights, n_labels,lambda/batchSize);
        weight[index]-=eta* deltaWeight;
    }
    
}



    //int block_size = 785;
    int n_blocks = 20;

    double* weight;
    int weight_size=(size_image+1)*10;
    cudaMallocHost((void**)&weight,weight_size*sizeof(double));
    psgd.testGPU(weight, testingData, testingLabels, n_images_test, size_image+1, 10);
    //update the weight
    for(int j=0;j<n_iterations;j++){
        updateWeightKernel<<<n_blocks, BLOCK_SIZE>>>(weight,trainingData,trainingLabel,eta,n_images,size_image+1,10,lambda);
        cudaDeviceSynchronize();
	}
        

    psgd.testGPU(weight, testingData, testingLabels, n_images_test, size_image+1, 10);




