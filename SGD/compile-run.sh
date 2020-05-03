#!/bin/bash
module purge
module load gcc-6.3.0
module load cuda-10.2
module list
rm -f hogwildSGD
nvcc -arch=compute_30 -o hogwildSGD hogwildSGD.cu -O3 -Xcompiler -fopenmp --maxrregcount 60 --expt-relaxed-constexpr;
sh hogwildSGD
