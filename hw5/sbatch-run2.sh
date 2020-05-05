#!/bin/sh
#

#SBATCH --verbose
#SBATCH --job-name=hw5
#SBATCH --output=slurm_hw5%j.out
#SBATCH --error=slurm_hw5%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB


#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end  # email me when the job ends

module purge
module load openmpi/gnu/4.0.2
make clean
make
mpirun -np 64 ssort 10000
mpirun -np 64 ssort 100000
mpirun -np 64 ssort 1000000
