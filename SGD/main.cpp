//
//  main.cpp
//  parallelSGD
//
//  Created by Yue Sun on 4/9/20.
//  Copyright © 2020 Yue Sun. All rights reserved.
//

#include <iostream>

#include "dataReader.h"
#include "PSGD.h"
#include "MultiLog.h"
#include "LossType.h"

int main(int argc, const char * argv[]) {
    // insert code here...
    mnist data;
    int n_images;
    int size_image;
    double **trainingData;
    trainingData=data.read_mnist_images("train-images.idx3-ubyte",n_images,size_image);
    
    printf("image size: %d\n",size_image+1);
    PSGD algo(4);
    algo.initWeight(size_image+1,10);
    
    vector<double> weight=algo.getWeight();
    
    MultiLog mlog;
    for(int i=300;i<303;i++){
        printf("%f ",weight[i]);
    }
    
    /*
    
    for(int i=0;i<5;i++){
        for(int j=0;j<size_image+1;j++){
            printf("%f ",trainingData[i][j]);
        }
        printf("\n");
    }
    
    */
    
    int n_labels;
    uchar *testingData;
    testingData=data.read_mnist_labels("train-labels.idx1-ubyte",n_labels);
    /*
    for(int i=0;i<5;i++){
        printf("%d ",testingData[i]);
    }
    */
    
    double lambda;
    lambda=0.001;
    //double loss;
    //loss=mlog.getLoss(weight,trainingData,testingData,n_images,size_image+1,10);
    //printf("%f ",loss)4
    printf("start update\n\n\n");
    algo.updateWeight(mlog,trainingData,testingData,lambda,5000,n_images,size_image+1,10);
    //updateWeight(LossType& loss,uchar** trainingData,uchar* testingData,double lambda,int n_iterations,int n_data,int size_weight,int size_label)
    weight=algo.getWeight();
    
    for(int i=300;i<303;i++){
        printf("%f ",weight[i]);
    }
    /*
    uchar a=1;
    int b=1;
    if(a==b){printf("true");}
    */
    for(int i=0;i<n_images;i++){
        free(trainingData[i]);
    }
    free(trainingData);
    free(testingData);
    return 0;
}
