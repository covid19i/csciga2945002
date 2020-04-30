/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N*N unknowns, each processor works with its
 * part, which has lN = N*N/p unknowns.
 * Author: Ilyeech Kishore Rapelli
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lNsqrt, double invhsq){
  int i, j;
  double tmp, gres = 0.0, lres = 0.0;

  for (i = 1; i <= lNsqrt; i++){
    for(j = 1; j <= lNsqrt; j++){
      tmp = ((4.0*lu[i*(lNsqrt+2) + j] - lu[(i-1)*(lNsqrt+2) + j] - lu[(i+1)*(lNsqrt+2) + j] - lu[i * (lNsqrt+2) + j-1] - lu[i * (lNsqrt+2) + j+1] ) * invhsq - 1);
      lres += tmp * tmp;
    }
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}


int main(int argc, char * argv[]){
  int mpirank, i, j, p, N, lN, iter, max_iters;
  MPI_Status status, status1, status2, status3;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  //printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

  sscanf(argv[1], "%d", &N);
  sscanf(argv[2], "%d", &max_iters);

  if(max_iters < 5){
    printf("Max Iterations is %d. It must be at least 5.\n", max_iters);
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  int  pnew = p;
  if(mpirank == 0)
  while(pnew > 0 && pnew != 1) {
    if(pnew % 4 != 0){
      printf("p: %d, pnew = p/4^k = %d, pnew % 4 = %d\n", p, pnew, pnew % 4);
      printf("Exiting. p must be a power of 4\n");
      MPI_Abort(MPI_COMM_WORLD, 0);
    } else {
      pnew = pnew/4;
    }
  }
  int Nsqrt = sqrt(N);
  int psqrt = sqrt(p);
  /* compute number of unknowns handled by each process */
  int lNsqrt = Nsqrt / psqrt;
  int mpirankx, mpiranky;
  if ((Nsqrt % psqrt != 0) && mpirank == 0 ) {
    printf("Nsqrt: %d, local Nsqrt: %d\n", Nsqrt, lNsqrt);
    printf("Exiting. Nsqrt must be a multiple of psqrt\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double * lu    = (double *) calloc(sizeof(double), (lNsqrt + 2)*(lNsqrt+2));
  double * lunew = (double *) calloc(sizeof(double), (lNsqrt + 2)*(lNsqrt+2));
  double * lutemp;
  /* Allocation of extra ghost vectors for copying boundary columns efficiently between processes */
  double * luleft_out = (double *) calloc(sizeof(double), (lNsqrt));
  double * luright_out = (double *) calloc(sizeof(double), (lNsqrt));
  double * luleft_in = (double *) calloc(sizeof(double), (lNsqrt));
  double * luright_in = (double *) calloc(sizeof(double), (lNsqrt));

  double h = 1.0 / (Nsqrt+ 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;

  /* initial residual */
  gres0 = compute_residual(lu, lNsqrt, invhsq);
  gres = gres0;

  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {
    /* Jacobi step for local points */
    for (i = 1; i <= lNsqrt; i++){
      for (j = 1; j <= lNsqrt; j++) {
        lunew[i*(lNsqrt+2)+j]  = 0.25 * (hsq + lu[(i - 1) * (lNsqrt+2) + j] + lu[(i + 1) * (lNsqrt+2) + j] + lu[i * (lNsqrt+2) + j-1] + lu[i * (lNsqrt+2) + j+1] );
      }
    }

    /* communicate ghost values */
    mpirankx = mpirank / psqrt;
    mpiranky = mpirank % psqrt;
    /*for(i=0; i< lNsqrt; i++){
      for(j = 0; j< lNsqrt;j++){
        printf("latest[%d][%d]: %f\n", mpirankx*(Nsqrt/psqrt) + i + 1, mpiranky*(Nsqrt/psqrt) + j + 1, lunew[(i+1) * (lNsqrt+2) + j + 1]);
      }
    }*/
    if (mpirankx < psqrt - 1) {
      /* If not in uppermost row of the processes, send/recv bdry values to the row above. Refer to the doc hpc20_assignment5.pdf */
      MPI_Send(&(lunew[(lNsqrt)*(lNsqrt+2)+1]), lNsqrt, MPI_DOUBLE, mpirank+psqrt, 124, MPI_COMM_WORLD);
      MPI_Recv(&(lunew[(lNsqrt+1)*(lNsqrt+2)+1]), lNsqrt, MPI_DOUBLE, mpirank+psqrt, 123, MPI_COMM_WORLD, &status);
    }
    if (mpirankx > 0) {
      /* If not in the bottommost row of the processes, the send/recv bdry values to the row below. */
      MPI_Send(&(lunew[1 + (lNsqrt +2)]), lNsqrt, MPI_DOUBLE, mpirank-psqrt, 123, MPI_COMM_WORLD);
      MPI_Recv(&(lunew[1]), lNsqrt, MPI_DOUBLE, mpirank-psqrt, 124, MPI_COMM_WORLD, &status1);
    }
    for(i = 1; i <= lNsqrt; i++) {
      luleft_out[i-1] = lunew[i*(lNsqrt+2)+1];
      luright_out[i-1] = lunew[i*(lNsqrt+2)+lNsqrt];
    }
    if (mpiranky < psqrt - 1) {
      /* Copying to another contiguous array to call MPI_Send just once */
      for(i = 1; i <= lNsqrt; i++) {
        luright_out[i-1] = lunew[i*(lNsqrt+2)+lNsqrt];
      }
      MPI_Send(&(luright_out[0]), lNsqrt, MPI_DOUBLE, mpirank+1, 137, MPI_COMM_WORLD);
      MPI_Recv(&(luright_in[0]), lNsqrt, MPI_DOUBLE, mpirank+1, 136, MPI_COMM_WORLD, &status2);
      for(i = 1; i <= lNsqrt; i++) {
        lunew[i*(lNsqrt+2)+lNsqrt+1] = luright_in[i-1];
      }
    }
    if (mpiranky > 0) {
      for(i = 1; i <= lNsqrt; i++) {
        luleft_out[i-1] = lunew[i*(lNsqrt+2)+1];
      }
      MPI_Send(&(luleft_out[0]), lNsqrt, MPI_DOUBLE, mpirank-1, 136, MPI_COMM_WORLD);
      MPI_Recv(&(luleft_in[0]), lNsqrt, MPI_DOUBLE, mpirank-1, 137, MPI_COMM_WORLD, &status3);
      for(i = 1; i <= lNsqrt; i++) {
        lunew[i*(lNsqrt+2)] = luleft_in[i-1];
      }
    }

    //Barrier not required as its blocking send and receive.
    //MPI_Barrier(MPI_COMM_WORLD);

    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    //if (0 == (iter % 1)){//(max_iters/5))) {
    if (0 == (iter % (max_iters/5)) || (iter == max_iters -1)) {
      gres = compute_residual(lu, lNsqrt, invhsq);
      if (0 == mpirank) {
	printf("Iter %d: Residual: %g\n", iter, gres);
      }
    }
  }

  /* Clean up */
  free(lu);
  free(lunew);
  free(luleft_out);
  free(luright_out);
  free(luleft_in);
  free(luright_in);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("\nWith %d processes, N = %d, N_l = %d: time elapsed is %f seconds.\n\n", p, N, N/p, elapsed);
  }
  MPI_Finalize();
  return 0;
}
