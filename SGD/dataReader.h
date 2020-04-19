//
//  dataReader.h
//  parallelSGD
//
//  Created by Yue Sun on 4/9/20.
//  Copyright © 2020 Yue Sun. All rights reserved.
//

#ifndef dataReader_h
#define dataReader_h

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

typedef unsigned char uchar;
using namespace std;
class mnist{
    
    
public:
    int reverseInt (int i)
    {
        unsigned char c1, c2, c3, c4;

        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;

        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    }
    
    
    double** read_mnist_images(string full_path, int& number_of_images, int& image_size)
    {
        
        ifstream file(full_path, ios::binary);

        if(file.is_open()) {
	    //printf("FILE OPEN: %s\n", &full_path);
            int magic_number = 0, n_rows = 0, n_cols = 0;

            file.read((char *)&magic_number, sizeof(magic_number));
            magic_number = reverseInt(magic_number);
	    //printf("MAGIC NUMBER:%d\n", magic_number);
            if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

            file.read((char *)&number_of_images, sizeof(number_of_images));
            number_of_images = reverseInt(number_of_images);
            //printf("number of images %d\n",number_of_images);
            file.read((char *)&n_rows, sizeof(n_rows));
            n_rows = reverseInt(n_rows);
            //printf("%d\n",n_rows);
            file.read((char *)&n_cols, sizeof(n_cols));
            n_cols = reverseInt(n_cols);
            //printf("%d\n",n_cols);

            image_size = n_rows * n_cols;

            double** _dataset = new double*[number_of_images];
            
            for(int i = 0; i < number_of_images; i++) {
                //_dataset[i] = new uchar[image_size+1];
                char* temp=(char*)malloc((image_size)*sizeof(char));
                _dataset[i] = (double*)malloc((image_size+1)*sizeof(double));
                
                file.read(temp, image_size);
                for(int j=0;j<image_size;j++){
		    _dataset[i][j]=temp[j]/255.0; // Is this giving rise to zeroes?
                }
                
                _dataset[i][image_size]=1;
                free(temp);
            }
	    
            return _dataset;
        }
        
        else {
            throw runtime_error("Unable to open file `" + full_path + "`!");
        }
        
    }
    
    
    
    
    
    uchar* read_mnist_labels(string full_path, int& number_of_labels)
    {

        ifstream file(full_path, ios::binary);

        if(file.is_open()) {
	    //printf("FILE OPEN: %s\n", full_path);
            int magic_number = 0;
            file.read((char *)&magic_number, sizeof(magic_number));
            magic_number = reverseInt(magic_number);

            if(magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

            file.read((char *)&number_of_labels, sizeof(number_of_labels));
            number_of_labels = reverseInt(number_of_labels);
            //printf("number of labels %d\n",number_of_labels);

            uchar* _dataset = (uchar*)malloc(number_of_labels*sizeof(uchar));
            for(int i = 0; i <number_of_labels; i++) {
                file.read((char*)&_dataset[i], 1);
            }
            return _dataset;
        }
        else{
            throw runtime_error("Unable to open file `" + full_path + "`!");
        }
    }
    
    
    
    
    
};





#endif /* dataReader_h */
