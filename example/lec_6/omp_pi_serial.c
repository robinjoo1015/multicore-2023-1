#include <omp.h>
#include <stdio.h>

long num_steps = 10000000; 
double step;

void main ()
{ 
	long i; double x, pi, sum = 0.0;
	step = 1.0/(double) num_steps;

	#pragma omp parallel for reduction(+:sum) private(x) 
	for (i=0;i< num_steps; i++){
		x = (i+0.5)*step;
		sum = sum + 4.0/(1.0+x*x);
	}
	pi = step * sum;
	printf("pi=%.8lf\n",pi);
}
