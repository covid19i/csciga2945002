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
    
    void updateWeight(LossType& loss,double** trainingData,uchar* testingData,double lambda,int n_iterations,int n_data,int n_weights,int n_labels){
        
        
        
        //printf("%d ",weight_size);
        
        
        //double* copy_weight=(double*)malloc(weight_size*sizeof(double));
        vector<double>copy_weight(weight_size);
        //double* parallel_weight;
        //double* delta_weight;
        for(int i=0;i<weight_size;i++)
        {
            copy_weight[i]=weight[i];
            weight[i]=0;
        }
        
        
        //#pragma omp parallel num_threads(n_threads) shared(copy_weight) private(parallel_weight,delta_weight)
        //#pragma omp for
        for(int n=0;n<n_threads;n++)
        {
            vector<double>parallel_weight(weight_size);
            vector<double>delta_weight(weight_size);

            for(int i=0;i<weight_size;i++)
            {
                parallel_weight[i]=copy_weight[i];//isnan
            }
            

            for(int j=0;j<n_iterations;j++)
            {
                    
                int index=rand() % n_data;//
                //printf("index %d ",index);
                //n_data is 60000, size_weights is 28*28+1, size_label is 10
                //printf("index %d",index);
                
                
                delta_weight=loss.getGradient(parallel_weight, trainingData[index], testingData[index], n_weights, n_labels);
                
                
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
                    parallel_weight[k]-=lambda*delta_weight[k];
                }
                //printf("delta_weight %f %f %f \n",parallel_weight[300],parallel_weight[301],parallel_weight[302]);
                
                
                
            }
                
            //printf("%d,delta_weight %f %f %f \n",omp_get_thread_num(),parallel_weight[300],parallel_weight[301],parallel_weight[302]);

            //#pragma omp critical
            {
                //printf("critical %d\n",omp_get_thread_num());
                for(int k=0;k<weight_size;k++){
                    weight[k]+=parallel_weight[k]/n_threads;
                }
                
            }
        }
        
    }
    
    ~PSGD(){}
};




#endif /* PSGD_h */
