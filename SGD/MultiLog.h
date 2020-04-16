//
//  MultuLog.h
//  parallelSGD
//
//  Created by Yue Sun on 4/11/20.
//  Copyright © 2020 Yue Sun. All rights reserved.
//
#include "LossType.h"
#include <math.h>
#ifndef MultiLog_h
#define MultiLog_h
typedef unsigned char uchar;
using namespace std;

class MultiLog: public LossType{
public:
    double getLoss(vector<double> weight,double** data,uchar* label,int size_data,int size_image,int size_label){
        
        double summ=0;
        for(int i=0;i<size_data;i++)
        {
            
            for(int j=0;j<size_image;j++)
            {
            
                double expSum=0;
                
                for(int k=0;k<size_label;k++)
                {
                    expSum += exp(weight[k*size_image+j] * data[i][j]);
                    if(k==label[i]){
                        summ += weight[k*size_image+j] * data[i][j];
                    }
                }
                summ-=log(expSum);
                
            }
            
            
        }
        
        return summ;
    }
    
    vector<double> getGradient(vector<double> weight,double* data,uchar label,int size_data,int size_label){
        
        vector<double> delta_weight(size_data*size_label);
        
        vector<double> probList(size_label);
        double probSum=0;
        
        /*
        for(int i=0;i<size_label;i++){
            for(int j=0;j<size_data;j++){
                if(isnan(weight[i*size_data+j])){
                    printf("%d,%d,%f\n",i,j,weight[i*size_data+j],data[i*size_data+j]);
                }
            }
        }
        */
        
        //calculate the probability
        for(int i=0;i<size_label;i++){
            double prob=0;
            
            for(int j=0;j<size_data;j++){
                
                prob+=weight[i*size_data+j]*data[i*size_data+j];
                //if(isnan(prob)){printf("%d,%d,%f,%f",i,j,weight[i*size_data+j],data[i*size_data+j]);exit(1);}
                
            }
            
            probList[i]=exp(prob);
            probSum+=exp(prob);
        }
        
        
        
        for(int i=0;i<size_label;i++){
            probList[i]/=probSum;
        }
        
        
        
        for(int i=0;i<size_label;i++){
            
            double sign=(i==label)?1:0;
            
            for(int j=0;j<size_data;j++){
                delta_weight[i*size_data+j]=-(sign-probList[i])*data[j];
            }
            
        }
        //printf("delta_weight inside gradient descent %f %f %f \n",delta[300],delta[301],delta[302]);
        return delta_weight;
    }
    //getGradient(parallel_weight, trainingData[index], testingData[index], size_weight, size_label)
    
};
#endif /* MultiLog_h */
