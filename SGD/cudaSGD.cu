//
//  cudaSGD.cpp
//  parallelSGD
//
//  Created by Yue Sun on 4/22/20.
//  Copyright © 2020 Yue Sun. All rights reserved.
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

__host__ __device__ double getOneGradient(double* weight,int index,const double*trainingData,const uchar* trainingLabel,double eta,int n_data,int n_weights,int n_labels,double lambda){
    //n_data is 2. which means i will use two data points to update one weight.
    
    double delta_weight=0;
    int i=index / n_weights;//i is for label i
    int j=index % n_weights;//j is for data j
    //printf("%d %d %d %d %d %d \n",index,i,j,n_weights,n_data,n_labels);
    
    double* probList;
    probList=(double*)malloc(n_labels*sizeof(double));
    for(int b=0;b<n_data;b++){
    
        double probSum=0;
        
        
        for(int l=0;l<n_labels;l++){
            double probExp=0;
            for(int w=0;w<n_weights;w++){
                //printf("%d %d %d %f %f \n",l,w,b,weight[l*n_weights+w],trainingData[b*n_weights+w]);
                probExp+=weight[l*n_weights+w]*trainingData[b*n_weights+w];
                
            }
            probList[l]=exp(probExp);
            probSum+=exp(probExp);
            
        }
        
        
        probList[i]/=probSum;
        //printf("probList[i]: %f \n",probList[i]);
        
        double sign = (trainingLabel[b]==i)?1:0;
        double partialDerivative = (sign-probList[i])*trainingData[b*n_weights+j];
        partialDerivative += lambda * 2 * weight[i*n_weights+j];
        delta_weight -= partialDerivative;
        
    }
    //printf("index: %d delta_weight: %f\n",index,delta_weight);
    free(probList);
    return delta_weight;
    
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
            //printf("%d ",label[b]);
            for(int w=0;w<n_weights;w++){
                data[b*n_weights+w]=trainingData[r*n_weights+w];
                //printf("%d %d %f \n",r,w,trainingData[r*n_weights+w]);
            }
        }
        //printf("%d %f \n",index,weight[index]);
        
        deltaWeight=getOneGradient(weight,index, data, label,eta, batchSize, n_weights, n_labels,lambda/batchSize);
        weight[index]-=eta* deltaWeight;
        
        //printf("%d %f %f \n",index,deltaWeight,weight[index]);
        
        free(data);
        free(label);
    }
    
}



int main(int argc, const char * argv[]) {
    // insert code here...
    mnist data;
    int n_images;
    int size_image;
    double **tempData;
    tempData = data.read_mnist_images("train-images.idx3-ubyte",n_images, size_image);
    //n_images 60000,size_images=785
    double * trainingData;
    cudaMallocHost((void**)&trainingData,n_images*(size_image+1)*sizeof(double));
    
    for(int i=0;i<n_images;i++){
        for(int j=0;j<size_image+1;j++){
            trainingData[i*(size_image+1)+j]=tempData[i][j];
        }
    }
    
    int n_labels;
    uchar *tempLabel;
    
    tempLabel = data.read_mnist_labels("train-labels.idx1-ubyte",n_labels);
    uchar* trainingLabel;
    cudaMallocHost((void**)&trainingLabel,n_labels*sizeof(uchar));
    for(int i=0;i<n_labels;i++){
        trainingLabel[i]=tempLabel[i];
    }
   
    int n_images_test;
    int size_image_test;
    double **testingData;
    testingData = data.read_mnist_images("t10k-images-idx3-ubyte", n_images_test, size_image_test);
    int n_labels_test;
    uchar *testingLabels;
    testingLabels = data.read_mnist_labels("t10k-labels-idx1-ubyte",n_labels_test);

    PSGD psgd(1);
    psgd.initialize(size_image+1,10);
    //vector<double> weight_openmp = psgd.getWeight();
 
    //define the size of cuda grid and block. after test, this is the maximum i can use
    dim3 gridSize(4,4);
    dim3 blockSize(5,5);
    
    
    double* weight;
    int weight_size=(size_image+1)*10;
    cudaMallocHost((void**)&weight,weight_size*sizeof(double));
    
    //initialize the weight
    int seed =1;//chrono::system_clock::now().time_since_epoch().count();
    default_random_engine generator (seed);
    normal_distribution<double> distribution (0.0,1.0);
    for (int i=0;i<weight_size;i++){
        weight[i]=distribution(generator);
    }
    psgd.testGPU(weight, testingData, testingLabels, n_images_test, size_image+1, 10);
    
    
    
    
    printf("Enter iterations (> 10):\n");
    int n_iterations=1000;
    scanf("%d", &n_iterations);
    
    double eta;
    eta=0.001;
    printf("\nEnter learning rate (eta = 0.001):\n");
    scanf("%lf", &eta);
    
    double lambda;
    lambda=0.001;
    printf("\nEnter regularization parameter (lambda = 0.001):\n");
    scanf("%lf", &lambda);
    
    double oldLoss=getLoss(weight,tempData,tempLabel,n_images,size_image+1,10,lambda);
    printf("old loss: %f \n",oldLoss);
    double t = omp_get_wtime();

    //update the weight
    int offset=0;
    for(int j=0;j<n_iterations;j++){
        offset=(j%20)*4*4*5*5;
        updateWeightKernel<<<gridSize,blockSize>>>(weight,trainingData,trainingLabel,eta,n_images,size_image+1,10,2,lambda,offset);
        cudaDeviceSynchronize();
    	if(j %(n_iterations/5) == 0 || j == n_iterations-1){
		double loss_now = getLoss(weight, tempData, tempLabel, n_images, size_image+1,10, lambda);
		printf("Training (log)loss: %f\t thread:%d\n",loss_now, omp_get_thread_num());
		psgd.testGPU(weight, testingData, testingLabels, n_images_test, size_image+1, 10);
	}
        
    }
    t = omp_get_wtime() - t;
    printf("Training ran for %10f\n", t);
    
    double newLoss=getLoss(weight,tempData,tempLabel,n_images,size_image+1,10,lambda);
    printf("new loss: %f \n",newLoss);

    MultiLog mlog;
    mlog.setLambda(lambda);//Regularization parameter
    psgd.loss = &mlog;//setting the loss function in PSGD object  

    psgd.testGPU(weight, testingData, testingLabels, n_images_test, size_image+1, 10);
 
    printf("end");
    free(tempData);
    free(tempLabel);
    cudaFree(trainingData);
    cudaFree(trainingLabel);
    cudaFree(weight);
    
    
    
    
}
