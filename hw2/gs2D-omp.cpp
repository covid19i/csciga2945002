//
//  gs2D-omp.cpp
//  
//
//  Created by Ilyeech Kishore Rapelli on 03/13/20.
//

#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
    long N=0;
    long Loops = 0;

    printf("Enter the no of points:");
    scanf("%ld", &N);
    printf("Enter maximum number of loops:");
    scanf("%ld", &Loops);
    double h = (double) 1/(N+1);
    //printf("\nN: %ld\th: %f\n", N, h);
    double normResidual = 0;
    double initialNormResidual = 0;
    
    double** f = (double **) malloc((N+2) * sizeof(double *));
    for(long j = 0; j <N+2; j++)
        f[j] = (double *) malloc((N+2) * sizeof(double));
    double** u1 = (double **) malloc((N+2) * sizeof(double *));
    for(long j = 0; j <N+2; j++)
        u1[j] = (double *) malloc((N+2) * sizeof(double));
    double** u2 = (double **) malloc((N+2) * sizeof(double *));
    for(long j = 0; j <N+2; j++)
        u2[j] = (double *) malloc((N+2) * sizeof(double));
    double** diff = (double **) malloc((N+2) * sizeof(double *));
    for(long j = 0; j <N+2; j++)
        diff[j] = (double *) malloc((N+2) * sizeof(double));
    double** temp;
    double** latest = u1;
    double** secondLatest = u2;

    long i = 0;
    double t;
    int nrT;
    double singleC = 1;
    int nrThreads[6] = {1, 2, 4, 8, 16, 32};
    for(int nrTIndex = 0; nrTIndex < 6; nrTIndex++) {
	nrT = nrThreads[nrTIndex];
	normResidual = 0;
	initialNormResidual = 0;
	i = 0;
	latest = u1;
	secondLatest = u2;
	
	#ifdef _OPENMP
	t = omp_get_wtime();
	#endif
	
	for(long j = 0; j < N+2; j++){
	    for(long k =0; k <N+2; k++){
		f[j][k] = 1; //drand48()-0.5;//f is the vector of all 1's
		//printf("f[j][k]=%f\n", f[j][k]);
		u1[j][k] = 0;//do remember the boundary points are always zero.
		u2[j][k] = 0;
		diff[j][k] = 0;//It's unnecessary though
	    }
	}
	
	for(i=0; i < Loops; i++){
	    #pragma omp parallel for num_threads(nrT) schedule(guided, 16)
	    for(long j=1; j<N+1;j++){
		for(long k=1; k < N+1; k++){
		    if(((int)(k-j))%2 == 0)//Red points
			    secondLatest[j][k]= 0.25 * (h*h*f[j][k] + latest[j-1][k] + latest[j+1][k] + latest[j][k+1] + latest[j][k-1]);
		    //Gauss-Seidel algo for Laplace equations
		}
	    }
	    #pragma omp parallel for num_threads(nrT) schedule(guided, 16)
	    for(long j=1; j<N+1;j++){
		for(long k=1; k < N+1; k++){
		    if(((int)(k-j))%2 != 0)//Black points
			    secondLatest[j][k]= 0.25 * (h*h*f[j][k] + secondLatest[j-1][k] + secondLatest[j+1][k] + secondLatest[j][k+1] + secondLatest[j][k-1]);
		}
	    }
	    temp=latest;//switching pointers
	    latest = secondLatest;//latest points to the latest values now
	    secondLatest = temp;//second latest points to the second latest values
	    
	    normResidual = 0;
	    #pragma omp parallel for num_threads(nrT) reduction(+:normResidual) schedule(guided, 16)
	    for(long j=1; j<N+1;j++){
		for(long k=1; k < N+1; k++){
		    diff[j][k] = (N+1)*(N+1)* (4*latest[j][k] - latest[j+1][k] - latest[j-1][k] - latest[j][k-1] - latest[j][k+1])
				- f[j][k];// a term in Au-f;
		    //printf("latest[j][k]: %f\t Norm Residual term: %f \t%ld\t%ld\n", latest[j][k], diff[j][k], j, k);
		    //printf("latest[%ld][%ld]: %f\t Norm Residual term: %10f\n", j, k, latest[j][k], diff[j][k]);

		    normResidual += diff[j][k]*diff[j][k];
		}
	    }
	    if(i==0){
		initialNormResidual = normResidual;
		//printf("Initial Norm Residual: %f\n", initialNormResidual);
	    } else {
	    //printf("%f\n", normResidual);
	    }
	    if(normResidual < initialNormResidual * 0.000001){
		//printf("\nExiting the loop in iteration %ld \n", i);
		break;
	    }
	}

	//printf("Initial Norm Residual: \t%f \n", initialNormResidual);
	//printf("Ending Norm Residual: \t%f\n", normResidual);
	//printf("The loop ran %ld times.\n", i);
	
	#ifdef _OPENMP
	t = omp_get_wtime() - t;
	#endif
	
	if(nrT == 1)
	  singleC = t;
	printf("%d: time elapsed = %f, speedup = %f\n", nrT, t, singleC/t);
	printf("%d: %f GB/s\n", nrT, 16*N*N*i*sizeof(double) / 1e9 / t);
	printf("%d: %f Gflops/s\n", nrT, 15*N*N*i / 1e9 / t);
    }
    
    for(long j = 0; j <N+2; j++){
	free(f[j]);
	free(u1[j]);
	free(u2[j]);
	free(diff[j]);
    }
    free(f);
    free(u1);
    free(u2);
    free(diff);
}
