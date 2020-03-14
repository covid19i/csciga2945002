/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

float a[VECLEN], b[VECLEN];

float dotprod ()
{
int i,tid;
float sum = 0;
//This sum variable is shared across the threads now as the parallel construct is
    //inside the dotprod()
#pragma omp parallel shared(sum) private(i, tid)
{
tid = omp_get_thread_num();
#pragma omp for reduction(+:sum)
  for (i=0; i < VECLEN; i++)
    {
    sum = sum + (a[i]*b[i]);
    printf("  tid= %d i=%d\n",tid,i,a[i]);
    }
}
    //The value has to be returned to be stored in the sum variable of main()
return sum;
}


int main (int argc, char *argv[]) {
int i;
float sum;

for (i=0; i < VECLEN; i++)
  a[i] = b[i] = 1.0 * i;
sum = 0.0;
//Parallelization didn't work as the sum variable inside the dotprod() was private to each thread.
    //OpenMP forbids that.
  sum = dotprod();

printf("Sum =  %f\n",sum);

}

