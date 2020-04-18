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
    trainingData = data.read_mnist_images("train-images.idx3-ubyte",n_images,
					  size_image);
    
    printf("image size+1: %d\nEnter no of threads:\n",size_image+1);
    int n_threads;
    scanf("%d", &n_threads);
    PSGD algo(n_threads);
    //size_image = 130;//REMOVE THIS AFTER TESTING
    algo.initWeight(size_image+1,10);
    
    vector<double> weight = algo.getWeight();
    
    for(int i = 300; i < 303; i++){
        printf("%f ",weight[i]);//REMOVE THIS AFTER TESTING
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
    uchar *testingData;//Should be RENAMED. This is training data (labels)
    testingData=data.read_mnist_labels("train-labels.idx1-ubyte",n_labels);
    MultiLog mlog;
    double oldloss = mlog.getLoss(weight,trainingData,testingData,n_images,
				  size_image+1,10);
    printf("\nold logloss: %f \n",oldloss);
    /*
    for(int i=0;i<5;i++){
        printf("%d ",testingData[i]);
    }
    */
    
    double lambda;
    lambda=0.001;
    //double loss;
    //loss=mlog.getLoss(weight,trainingData,testingData,n_images,size_image+1,10);
    //printf("%f ",loss)
    printf("start update\n\nEnter iterations for each thread:\n");
    int n_iter;
    scanf("%d", &n_iter);
    algo.updateWeight(mlog,trainingData,testingData,lambda,n_iter,n_images,
		      size_image+1,10);
    //updateWeight(LossType& loss,uchar** trainingData,uchar* testingData,
    //double lambda,int n_iterations,int n_data,int size_weight,int size_label)
    printf("Training complete.\n");
    weight=algo.getWeight();
    double newloss=mlog.getLoss(weight,trainingData,testingData,n_images,
				size_image+1,10);
    printf("new logloss: %f\n",newloss);
    /*
    for(int i=0;i<(size_image+1)*10;i++){
        printf("%f ",weight[i]);
    }
    */
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
