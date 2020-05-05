// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>

int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
    
  int N = 10000;
    sscanf(argv[1], "%d", &N);

  int* vec = (int*)malloc(N*sizeof(int));
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  //printf("rank: %d, first entry: %d\n", rank, vec[0]);

  // sort locally
  std::sort(vec, vec+N);

  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector
    int* local_splitters = (int*) calloc(p-1, sizeof(int));
    int slide = (int) N/p;
    for(int i=0; i<p-1;i++){
	local_splitters[i] = vec[slide*(i+1)];
	//printf("%d-th Splitter at rank %d: %d\n", i, rank, local_splitters[i]);
    }
    
  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
    int root = 0;
    int *splitters_root;
    if(rank == root){
	splitters_root = (int*) calloc(p * (p-1), sizeof(int));
    }
    MPI_Gather(local_splitters, p-1, MPI_INT, splitters_root, p-1, MPI_INT, root, comm);

  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
    int *splitters_chosen = (int*) calloc((p-1), sizeof(int));
    if(rank == root){
	std::sort(splitters_root, splitters_root + (p*(p-1)) );
	for (int i=0; i < p-1;i++){
	    splitters_chosen[i]=splitters_root[(i+1)*(p-1)];
	    //printf("%d-th Splitter at root: %d\n", i, splitters_chosen[i]);
	}
    }
    
  // root process broadcasts splitters to all other processes
    MPI_Bcast(splitters_chosen, p-1, MPI_INT, root, comm);
    //printf("1-th Splitter at rank %d: %d\n", rank, splitters_chosen[1]);
  // every process uses the obtained splitters to decide which
  // integers need to be sent to which other process (local bins).
  // Note that the vector is already locally sorted and so are the
  // splitters; therefore, we can use std::lower_bound function to
  // determine the bins efficiently.
    int* sdispls = (int*) calloc(p, sizeof(int));
    int* rdispls = (int*) calloc(p, sizeof(int));
    int* sendcounts = (int*) calloc (p, sizeof(int));
    int* recvcounts = (int*) calloc (p, sizeof(int));
    sdispls[0] = 0;
    for(int i=1; i<p;i++){
	//The position of i-th splitter in the local sorted array
	sdispls[i] = std::lower_bound(vec, vec+N, splitters_chosen[i-1] + 1) - vec;
	sendcounts[i-1] = sdispls[i] - sdispls[i-1];
    }
    sendcounts[p-1] = N - sdispls[p-1];//total sendcounts = N
    int temp = 0;
    for(int i=0; i<p;i++){
	//The position of i-th splitter in the local sorted array
	temp += sendcounts[i];
    }
    //printf("Total sendcount: %d\t 3-th sendcount at rank %d: %d\n", temp, rank, sendcounts[3]);
    
    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, comm);
    temp = 0;
    for(int i=0; i<p;i++){
	//The position of i-th splitter in the local sorted array
	temp += recvcounts[i];
    }
    //printf("Total recvcount: %d\t 3-th recvcount at rank %d: %d\n", temp, rank, recvcounts[3]);

    rdispls[0] = 0;
    for(int i=1; i<p;i++)
	rdispls[i] = rdispls[i-1] + recvcounts[i-1];
    int recv_vec_length = rdispls[p-1] + recvcounts[p-1];
    printf("Total recv_vec_length: %d at rank %d\n", recv_vec_length, rank);
    int* recv_vec = (int*)malloc(recv_vec_length * sizeof(int));
    MPI_Alltoallv(vec, sendcounts, sdispls, MPI_INT, recv_vec, recvcounts, rdispls, MPI_INT, comm);
  //
  // Hint: the MPI_Alltoallv exchange in the next step requires
  // send-counts and send-displacements to each process. Determining the
  // bins for an already sorted array just means to determine these
  // counts and displacements. For a splitter s[i], the corresponding
  // send-displacement for the message to process (i+1) is then given by,
  // sdispls[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;

  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  // MPI_Alltoallv to exchange the data

  // do a local sort of the received data
    std::sort(recv_vec, recv_vec+recv_vec_length);
    printf("rank: %d, first entry: %d\n", rank, recv_vec[0]);
  // every process writes its result to a file
    {
      FILE* fd = NULL;
      char filename[256];
      snprintf(filename, 256, "output%02d.txt", rank);
      fd = fopen(filename,"w+");
      if(NULL == fd) {
	printf("Error opening file \n");
	return 1;
      }
      //fprintf(fd, "rank %d received from %d the message:\n", rank, origin);
      for(int n = 0; n < recv_vec_length; ++n)
	fprintf(fd, "%d\n", recv_vec[n]);

      fclose(fd);
    }

    free(vec);
    free(local_splitters);
    if(rank == root){
	free(splitters_root);
    }
    free(splitters_chosen);
    free(sdispls);
    free(rdispls);
    free(sendcounts);
    free(recvcounts);
    free(recv_vec);
  MPI_Finalize();
  return 0;
}
