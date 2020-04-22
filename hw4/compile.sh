#!/bin/bash
module purge
module load gcc-6.3.0
module load cuda-10.2
module list
rm hw4-1a
rm hw4-1b
rm hw4-2
nvcc -arch=sm_30 -o hw4-1a hw4-1a.cu -Xcompiler -fopenmp
nvcc -arch=sm_30 -o hw4-1b hw4-1b.cu -Xcompiler -fopenmp
nvcc -arch=sm_30 -o hw4-2 hw4-2.cu -Xcompiler -fopenmp
./hw4-1a
./hw4-1b
./hw4-2
