#!/bin/sh
#

#SBATCH --verbose
#SBATCH --job-name=hw5_jacobi_mpi
#SBATCH --output=slurm_hw5%j.out
#SBATCH --error=slurm_hw5_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=8-32
#SBATCH --ntasks-per-node=10
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB


#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end  # email me when the job ends

rm mpi-hw5-1
module purge
module load openmpi/gnu/4.0.2
mpic++ -O3 -o mpi-hw5-1 hw5-1.cpp
mpirun -np 4 mpi-hw5-1  40000 10000
mpirun -np 16 mpi-hw5-1 160000 10000
mpirun -np 64 mpi-hw5-1 640000 10000
mpirun -np 256 mpi-hw5-1 2560000 10000
mpirun -np 1 mpi-hw5-1 1000000 10000
mpirun -np 4 mpi-hw5-1 1000000 10000
mpirun -np 16 mpi-hw5-1 1000000 10000
mpirun -np 64 mpi-hw5-1 1000000 10000
mpirun -np 256 mpi-hw5-1 1000000 10000
