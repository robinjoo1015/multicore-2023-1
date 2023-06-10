#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

long num_steps = 10000000;
double step;

int main(int argc, char *argv[])
{
	long i;
	double x, pi, sum = 0.0;
	double start_time, end_time;

	int scheduling_type, chunk_size, num_thread;

	if (argc != 4)
	{
		printf("Wrong argunment, please enter scheduling_type_# chunk_size #_of_thread\n");
		return 0;
	}

	// parse arguments
	scheduling_type = atoi(argv[1]);
	chunk_size = atoi(argv[2]);
	num_thread = atoi(argv[3]);

	if (scheduling_type < 1 || scheduling_type > 3)
	{
		printf("Wrong scheduling_type_#, please enter between 1~3\n");
		return 0;
	}
	if (num_thread < 1)
	{
		printf("Wrong #_of_thread, please enter positive integer\n");
		return 0;
	}

	// set number of threads
	omp_set_num_threads(num_thread);
	// set scheduling type and chunk size
	omp_set_schedule(scheduling_type, chunk_size);

	start_time = omp_get_wtime();

	step = 1.0 / (double)num_steps;

	// parallelize, critical section for sum
#pragma omp parallel for reduction(+ : sum) default(shared) private(i, x) schedule(runtime)
	for (i = 0; i < num_steps; i++)
	{
		x = (i + 0.5) * step;
		sum = sum + 4.0 / (1.0 + x * x);
	}

	pi = step * sum;

	end_time = omp_get_wtime();
	double timeDiff = (end_time - start_time) * 1000;
	printf("Execution Time : %lfms\n", timeDiff);

	printf("pi=%.24lf\n", pi);
}

// typedef enum omp_sched_t {
//     omp_sched_static = 1,
//     omp_sched_dynamic = 2,
//     omp_sched_guided = 3,
//     omp_sched_auto = 4
// } omp_sched_t;
// omp_set_schedule(omp_sched_t kind, int chunk_size)