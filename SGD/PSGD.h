//
//  PSGD.h
//  parallelSGD
//
//  Created by Yue Sun on 4/10/20.
//  Copyright © 2020 Yue Sun. All rights reserved.
//
#include <iostream>
#include <cstdlib>
#include <string>
#include <random>
#include <chrono>
#include "LossType.h"
#include <omp.h>

#ifndef PSGD_h
#define PSGD_h
using namespace std;
typedef unsigned char uchar;

class PSGD{
private:
    vector<double> weight;
    int n_threads;
    int weight_size;
public:
    PSGD(int number_threads){
        n_threads=number_threads;
    }
    
    int getn_threads(){
	return n_threads;
    }
    
    void initWeight(int n_weights,int n_labels){
        
        int seed =1;//chrono::system_clock::now().time_since_epoch().count();
        default_random_engine generator (seed);
        normal_distribution<double> distribution (0.0,1.0);
        weight_size=n_weights*n_labels;
        
        weight.resize(weight_size);
        
        for(int i=0;i<weight_size;i++){
            weight[i]=distribution(generator);
        }
    }
    
    vector<double> getWeight(){
        return weight;
    }
    
    void updateWeight(LossType& loss,double** trainingData,uchar* testingData,double eta,int n_iterations,int n_data,int n_weights,int n_labels)
    {
        //printf("%d ",weight_size);
        //double* copy_weight=(double*)malloc(weight_size*sizeof(double));
        vector<double>copy_weight(weight_size);
	//vector<double>parallel_weight(weight_size);
	//vector<double>delta_weight(weight_size);
        for(int i=0;i<weight_size;i++)
        {
            copy_weight[i]=weight[i];
            weight[i]=0;
        }
        
        
        #pragma omp parallel num_threads(n_threads) shared(copy_weight)
        {
            #pragma omp for
            for(int n=0;n<n_threads;n++)
            {
                vector<double>parallel_weight(weight_size);
                vector<double>delta_weight(weight_size);

                for(int i=0;i<weight_size;i++)
                {
		    //Why is copy_weight is shared?
                    parallel_weight[i]=copy_weight[i];//isnan
                }

		double accum = 0;
                for(int j=0;j<n_iterations;j++)
                {
		    //printf("Thr %d, Iter %d started\n", n, j);
                    int index = rand() % n_data;//
                    //printf("index %d ",index);
                    //n_data is 60000, size_weights is 28*28+1, size_label is 10
                    //printf("index %d",index);
		    //del w = -grad f. But, because of a minus when we update weights, this MISTAKE is fine.
                    delta_weight=loss.getGradient(parallel_weight, trainingData[index], testingData[index], n_data, n_weights, n_labels);
		    if(j %(n_iterations/5) == 0 || j == n_iterations-1){
			// l2-norm
			accum = 0;
			for (int l = 0; l < weight_size; l++) {
			    accum += delta_weight[l] * delta_weight[l];
			}
			printf("Delta Norm[%d] \t= %f \tin thread %d\n", index, sqrt(accum), omp_get_thread_num());//NOT SURE if n can be used as the thread ID
		    }
		    //printf("Thr %d, Iter %d middle\n", n, j);
                    /*
                    for(int i=0;i<weight_size;i++)
                    {
                        if(isnan(delta_weight[i]))
                        {
                            printf("%f,%d,%d",delta_weight[i],j,i);
                            exit(1);
                        }
                    }
                    */
                    for(int k=0;k<weight_size;k++){
                        parallel_weight[k] -= eta*delta_weight[k];
                    }
		    if(j %(n_iterations/5) == 0 || j == n_iterations-1){
			// l2-norm
			accum = 0;
			for (int l = 0; l < weight_size; ++l) {
			    accum += parallel_weight[l] * parallel_weight[l];
			}
			printf("Weight Norm \t= %f \tin thread %d\n", sqrt(accum), omp_get_thread_num());
		    }
		    
		    //printf("Thr %d, Iter %d ended\n", n, j);
                    //printf("delta_weight %f %f %f \n",parallel_weight[300],parallel_weight[301],parallel_weight[302]);
                }
		//printf("%d,delta_weight %f %f %f \n",omp_get_thread_num(),parallel_weight[300],parallel_weight[301],parallel_weight[302]);
		#pragma omp critical
		{
		//printf("critical %d\n",omp_get_thread_num());
		    for(int k=0;k<weight_size;k++){
			weight[k] += parallel_weight[k]/n_threads;
		    }
		}
            }
        }
	
    }
    
    void test(double** testingData, uchar* testingLabels, int n_data, int n_weights,int n_labels)
    {
	int correct_data = 0;
        vector<double> probList(n_labels);
	double prob_exponent, maxProb, probSum;
	#pragma omp parallel num_threads(n_threads) reduction(+:correct_data)
        {
            #pragma omp for
	    for(int j=0; j<n_data; j++)
	    {
		//n_data is 60000, size_weights is 28*28+1, size_label is 10
		maxProb = 0;
		probSum = 0;
		for(int i=0;i<n_labels;i++){
		    //probList[i] = 0;//unnecessary
		    prob_exponent=0;//necessary
		    for(int k=0;k<n_weights;k++){
			prob_exponent += weight[i*n_weights+k]*testingData[j][k];
		    }
		    probList[i] = exp(prob_exponent);
		    if(probList[i] > maxProb)
			maxProb = probList[i];
		    probSum += probList[i];
		}
		if(probList[testingLabels[j]] == maxProb)
		    correct_data++;
		
	    }
        }
	printf("\n%d correct out of %d.\nRatio: %f\n", correct_data, n_data, (float)correct_data/n_data);
    }
    
    ~PSGD(){}
};

#endif /* PSGD_h */
