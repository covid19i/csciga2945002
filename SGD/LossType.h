//
//  LossType.h
//  parallelSGD
//
//  Created by Yue Sun on 4/12/20.
//  Copyright © 2020 Yue Sun. All rights reserved.
//

#ifndef LossType_h
#define LossType_h
typedef unsigned char uchar;
using namespace std;

class LossType{
public:
    virtual double getLoss(double* weight,double** data,uchar* label,int size_data,int size_weight,int size_label)=0;
    
    virtual vector<double> getGradient(vector<double> weight,double* data,uchar label,int size_data,int size_label)=0;
    //virtual double* getGradient(double* weight,uchar* data,uchar* label,int size_data,int size_label);
};

#endif /* LossType_h */
