#!/bin/sh
#

#SBATCH --verbose
#SBATCH --job-name=hw5_jacobi_mpi
#SBATCH --output=hw5__slurm_%j.out
#SBATCH --error=hw5_slurm_%j.err
#SBATCH --time=00:04:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB


#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end  # email me when the job ends

rm mpi-hw5-1
module load openmpi/gnu/4.0.2
mpic++ -O3 -o mpi-hw5-1 hw5-1.cpp
mpirun -np 16 mpi-hw5-1 160000 10000
