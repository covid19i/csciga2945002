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
    //trainingData = data.read_mnist_images("train-images-idx3-ubyte",n_images, size_image);
    trainingData = data.read_mnist_images("train-images.idx3-ubyte",n_images, size_image);
    int n_labels;
    uchar *trainingLabels;
    trainingLabels = data.read_mnist_labels("train-labels.idx1-ubyte",n_labels);

    /*printf("\n");
    for(int i=100;i<101;i++){
        for(int j=188;j<191;j++){
	    //printf("%f\t",trainingData[i][j]);
            printf("pixel[%d][%d]: %f\t",i, j, trainingData[i][j]);
        }
        printf("\n");
    }*/

    int n_images_test;
    int size_image_test;
    double **testingData;
    testingData = data.read_mnist_images("t10k-images-idx3-ubyte", n_images_test, size_image_test);
    int n_labels_test;
    uchar *testingLabels;
    testingLabels = data.read_mnist_labels("t10k-labels-idx1-ubyte",n_labels_test);

    //printf("image (roaster) size + 1: %d\n", size_image+1);
    printf("Enter no of threads:\n");
    int n_threads = 1;
    scanf("%d", &n_threads);

    PSGD psgd(n_threads);
    psgd.initialize(size_image+1,10);
    psgd.test(testingData, testingLabels, n_images_test, size_image+1, 10);
    vector<double> weight = psgd.getWeight();
    printf("\n");
    for(int i = 300; i < 303; i++){
        printf("weight[%d]: %f\t", i, weight[i]);
    }
    
    MultiLog mlog;
    double lambda;
    lambda=0.00;//0.001;
    printf("\nEnter regularization parameter (lambda = %f): (Note: make this zero if you don't want regularization.)\n", lambda);
    scanf("%lf", &lambda);
    mlog.setLambda(lambda);//Regularization parameter
    psgd.loss = &mlog;//setting the loss function in PSGD object
    double oldloss = mlog.getLoss(weight, trainingData, trainingLabels, n_images,
				  size_image+1,10);
    printf("\nold training logloss: %f\n",oldloss);
    /*
    for(int i=0;i<5;i++){
        printf("%d ",trainingLabels[i]);
    }
    */
    
    double eta;
    eta=0.001;
    //printf("\nEnter learning rate (eta = 0.001):\n");
    //scanf("%lf", &eta);
    //double loss;
    //loss=mlog.getLoss(weight,trainingData,trainingLabels,n_images,size_image+1,10);
    //printf("%f ",loss)
    //printf("Start Training.\n");
    printf("Enter iterations for each thread (> 10):\n");
    int n_iter = 10000;
    scanf("%d", &n_iter);
    
    double t = omp_get_wtime();
    psgd.train(mlog, trainingData, trainingLabels, eta,
    		n_iter, n_images, size_image+1, 10,
			testingData, testingLabels, n_images_test);//Training the model
    t = omp_get_wtime() - t;
    //updateWeight(LossType& loss,uchar** trainingData,uchar* trainingLabels,
    //double lambda,int n_iterations,int n_data,int size_weight,int size_label)
    weight = psgd.getWeight();
    /*printf("\n");
    for(int i = 300; i < 303; i++){
        printf("weight[%d]: %f\t", i, weight[i]);//REMOVE THIS AFTER TESTING
    }*/
    //printf("\nTraining complete.\n");
    weight = psgd.getWeight();
    double newloss = mlog.getLoss(weight, trainingData, trainingLabels, n_images, size_image+1, 10);
    printf("New training logloss: %f\n",newloss);
    //printf("Old training logloss: %f\n",oldloss);
    //printf("Diff training logloss: %f\n", newloss - oldloss);
    /*
    for(int i=0;i<(size_image+1)*10;i++){
        printf("%f ",weight[i]);
    }
    */
    
    //Testing the model
    printf("\nNo of iterations for each thread: %d\n", n_iter);
    printf("No of threads: %d\n", psgd.getn_threads());
    printf("Lambda (Regularization Parameter): %lf\n", mlog.getLambda());
    printf("Eta (Learning Rate): %lf\n", eta);
    psgd.test(testingData, testingLabels, n_images_test, size_image+1, 10);
    //psgd.test(trainingData, trainingLabels, n_images, size_image+1, 10);
    printf("\nTime elapsed in training = %f sec\n", t);
    printf("Time elapsed in training per iteration = %f sec\n", t/n_iter);
    
    /*
    uchar a=1;
    int b=1;
    if(a==b){printf("true");}
    */
    for(int i=0;i<n_images;i++){
        free(trainingData[i]);
    }
    for(int i=0;i<n_images_test;i++){
        free(testingData[i]);
    }
    free(trainingData);
    free(trainingLabels);
    free(testingData);
    free(testingLabels);
    return 0;
}
