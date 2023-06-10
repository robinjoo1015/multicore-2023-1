#include <stdio.h>
#include <omp.h>

int main()
{
	omp_set_num_threads(4);
	printf("before omp sections\n");
#pragma omp parallel sections
//#pragma omp parallel 
//	#pragma omp sections
	{
		#pragma omp section
		{
			printf("section 1: %d/%d\n",omp_get_thread_num(),omp_get_num_threads());
		}

		#pragma omp section
		{
			printf("section 2: %d/%d\n",omp_get_thread_num(),omp_get_num_threads());
		}
	}
	printf("after omp sections\n");
	return 0;
}
