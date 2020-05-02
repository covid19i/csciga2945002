#!/bin/bash
module purge
module load gcc-6.3.0
module load cuda-10.2
module list
rm hogwildSGD
nvcc -arch=sm_30 -o hogwildSGD hogwildSGD.cu -Xcompiler -fopenmp
./hogwildSGD
