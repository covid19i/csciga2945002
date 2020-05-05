module purge
module load mpi/openmpi-x86_64
rm mpi-hw5-1
make clean
make
mpirun -np 4 mpi-hw5-1  40000 10000
mpirun -np 16 mpi-hw5-1 160000 10000
mpirun -np 64 mpi-hw5-1 640000 10000
mpirun -np 256 mpi-hw5-1 2560000 10000
mpirun -np 1 mpi-hw5-1 1000000 10000
mpirun -np 4 mpi-hw5-1 1000000 10000
mpirun -np 16 mpi-hw5-1 1000000 10000
mpirun -np 64 mpi-hw5-1 1000000 10000
mpirun -np 256 mpi-hw5-1 1000000 10000
