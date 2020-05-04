#!/bin/sh
#

#SBATCH --verbose
#SBATCH --job-name=SGD_tf
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2GB


#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end  # email me when the job ends

module purge
module load tensorflow/python2.7/1.5.0
module list
python nikhil-mnist.py
