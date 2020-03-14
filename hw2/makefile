CC = g++
all: MMult1 val_test01_solved val_test02_solved omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 jacobi2D-omp gs2D-omp

MMult1 : MMult1.cpp
	$(CC) -O3 -fopenmp -march=native MMult1.cpp -o MMult1

val_test01_solved : val_test01_solved.cpp
	$(CC) -g -fopenmp -march=native val_test01_solved.cpp -o val_test01_solved

val_test02_solved : val_test02_solved.cpp
	$(CC) -g -fopenmp -march=native val_test02_solved.cpp -o val_test02_solved

omp_solved2 : omp_solved2.c
	$(CC) -fopenmp -O2 -march=native omp_solved2.c -o omp_solved2

omp_solved3 : omp_solved3.c
	$(CC) -fopenmp -O2 -march=native omp_solved3.c -o omp_solved3

omp_solved4 : omp_solved4.c
	$(CC) -fopenmp -O2 -march=native omp_solved4.c -o omp_solved4

omp_solved5 : omp_solved5.c
	$(CC) -fopenmp -O2 -march=native omp_solved5.c -o omp_solved5

omp_solved6 : omp_solved6.c
	$(CC) -fopenmp -O2 -march=native omp_solved6.c -o omp_solved6

jacobi2D-omp : jacobi2D-omp.cpp
	$(CC) -fopenmp -O3 -march=native jacobi2D-omp.cpp -o jacobi2D-omp

gs2D-omp : gs2D-omp.cpp
	$(CC) -fopenmp -O3 -march=native gs2D-omp.cpp -o gs2D-omp

.PHONY: clean
clean:
	rm MMult1 val_test01_solved val_test02_solved omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 jacobi2D-omp gs2D-omp
