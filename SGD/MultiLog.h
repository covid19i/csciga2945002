//
//  MultiLog.h
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
private:
    double lambda;//Regularization parameter for L2 Regularization
public:
    void setLambda(double l){
	lambda = l;
    }
    double getLambda(){
	return lambda;
    }
    double getLoss(vector<double> weight,double** data,uchar* label,int size_data,int size_weights,int size_label){
	//size_data = 3;
        //double summ = -1529628.136844;
        double summ=0;
	vector<double> exponent(size_label);
        for(int i=0; i<size_data; i++){
            /*for(int j=0; j<size_weights; j++){
                double expSum=0;
                for(int k=0; k<size_label; k++){
                    expSum += exp(weight[k*size_weights+j] * data[i][j]);
                    if(k==label[i]){
                        summ += weight[k*size_weights+j] * data[i][j];
                    }
                }
		if(expSum > 0){
		    summ -= log(expSum);
		}
            }*/
	    double expSum=0;
	    for(int k=0; k<size_label; k++){
		exponent[k] = 0;
                for(int j=0; j<size_weights; j++){
		    exponent[k] += weight[k*size_weights+j] * data[i][j];
                    if(k==label[i]){
                        summ -= exponent[k];
                    }
                }
		expSum += exp(exponent[k]);
            }
	    if(expSum > 0){
		summ += log(expSum);
	    }
        }
	
	double regularizationTerm = 0;
	for(int k=0; k<size_label; k++){
	    for(int j=0; j<size_weights; j++){
		regularizationTerm += weight[k*size_weights+j] * weight[k*size_weights+j];
	    }
	}
	regularizationTerm *= lambda;
	
        return summ + regularizationTerm;
    }
    
    vector<double> getGradient(vector<double> weight,double* data,uchar label,int n_data, int size_weights,int size_label){
        vector<double> delta_weight(size_weights*size_label);
        vector<double> probList(size_label);
        double probSum=0;
        //https://houxianxu.github.io/2015/04/23/logistic-softmax-regression/
        //calculate the probabilities for this datum for each class
        for(int i=0;i<size_label;i++){
            double prob_exponent=0;
            for(int j=0;j<size_weights;j++){
                prob_exponent += weight[i*size_weights+j]*data[j]; //if(isnan(prob)){printf("%d,%d,%f,%f",i,j,weight[i*size_data+j],data[i*size_data+j]);exit(1);}
            }
            probList[i] = exp(prob_exponent);
            probSum += exp(prob_exponent);
        }
	if(probSum <= 0){
	    printf("\nERROR in prob sum\n");//THIS HAS TO BE REMOVED? It can go to zero due to hardware!!!!!!!!!!!!!!!!!!!!!!!!
	    exit(1);
	}
        for(int i=0;i<size_label;i++){
            probList[i] /= probSum;
            double sign=(i==label)?1:0;
            for(int j=0;j<size_weights;j++){
                delta_weight[i*size_weights+j] = -(sign-probList[i]) * data[j];
		delta_weight[i*size_weights+j] -= (2/n_data) * lambda * delta_weight[i*size_weights+j];//scaling because in SGD, gradient is calculated for just one point.
            }
        }
        //printf("delta_weight inside gradient descent %f %f %f \n",delta[300],delta[301],delta[302]);
	
        return delta_weight;
    }
    //getGradient(parallel_weight, trainingData[index], testingData[index], size_weight, size_label)
};
#endif /* MultiLog_h */
