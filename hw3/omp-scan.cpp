#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
    //Changed this implementation to calculate the scan value at the last index also
  prefix_sum[0] = A[0];
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
    if (n == 0) return;
    prefix_sum[0] = A[0];
    int nrT = 8;
    printf("Number of threads: %d\n", nrT);
    long blockSize = (long) n/nrT;
    #pragma omp parallel for num_threads(nrT)
    for(long j = 1; j <= nrT; j++){
	prefix_sum[(j-1)*blockSize] = A[(j-1)*blockSize];
	long endOfBlock = std::min(n, j*blockSize);
	for(long k= 1 + (j-1)*blockSize; k<endOfBlock; k++){
	    prefix_sum[k]=prefix_sum[k-1] + A[k];
	}
	//printf("Inside block Offset: %ld at index: %ld\n", prefix_sum[endOfBlock-1], endOfBlock - 1);
    }
    
    //Serial offset calculation
    long* offset = (long*) malloc(nrT*sizeof(long));
    offset[0] = 0;
    for(long j = 2; j <= nrT; j++){
	offset[j-1] = prefix_sum[(j-1) * blockSize - 1] + offset[j-2];
    }
    
    //parallel correction using the above calculated offsets
    #pragma omp parallel for num_threads(nrT)
    for(long j = 2; j <= nrT; j++){
	//printf("Offset: %ld at index: %ld\n", offset[j-1], (j-1) * blockSize - 1);
	long endOfBlock = std::min(n, j*blockSize);
	for(long k=(j-1)*blockSize; k<endOfBlock; k++){
	    prefix_sum[k]=prefix_sum[k] + offset[j-1];
	}
    }
    
    /*//Serial code
    long offset = 0;
    for(long j = 2; j <= nrT; j++){
	offset = prefix_sum[(j-1) * blockSize - 1];
	//printf("Offset: %ld at index: %ld\n", offset, (j-1) * blockSize - 1);
	long endOfBlock = std::min(n, j*blockSize);
	for(long k=(j-1)*blockSize; k<endOfBlock; k++){
	    prefix_sum[k]=prefix_sum[k] + offset;
	}
    }*/
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
