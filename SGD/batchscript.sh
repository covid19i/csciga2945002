#!/bin/sh
#

#SBATCH --verbose
#SBATCH --job-name=Hogwild_SGD_Project
#SBATCH --output=slurm_hw5%j.out
#SBATCH --error=slurm_hw5_%j.err
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2GB


#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end  # email me when the job ends

rm hogwildSGD
module purge
module load gcc/6.3.0
module load cuda/10.2.89
module list
nvcc -arch=sm_30 -o hogwildSGD hogwildSGD.cu
./hogwildSGD

