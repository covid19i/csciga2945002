#!/bin/sh
#

#SBATCH --verbose
#SBATCH --job-name=compare_tf_cuda
#SBATCH --output=cuda_tf_slurm_%j.out
#SBATCH --error=cuda_tf_slurm_%j.err
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --mem=2GB


#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end  # email me when the job ends

module purge
module load tensorflow/python2.7/1.5.0
module load gcc/6.3.0
module list
python ./TensorFlow/nikhil-mnist.py
nvcc -arch=compute_30 -o hogwildSGD hogwildSGD.cu -Xcompiler -fopenmp --maxrregcount 60 --expt-relaxed-constexpr;
./hogwildSGD
