#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

int num_end = 200000;

// function to check whether the number is prime
bool is_prime(int x)
{
    int i;
    if (x <= 1)
    {
        return false;
    }
    for (i = 2; i < x; i++)
    {
        if (x % i == 0)
        {
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[])
{
    int i, sum = 0;
    double start_time, end_time;

    int scheduling_type, num_thread;

    if (argc != 3)
    {
        printf("Wrong argunment, please enter scheduling_type_# #_of_thread\n");
        return 0;
    }

    // parse arguments
    scheduling_type = atoi(argv[1]);
    num_thread = atoi(argv[2]);

    if (scheduling_type < 1 || scheduling_type > 4)
    {
        printf("Wrong scheduling_type_#, please enter between 1~4\n");
        return 0;
    }
    if (num_thread < 1)
    {
        printf("Wrong #_of_thread, please enter positive integer\n");
        return 0;
    }

    // set number of threads
    omp_set_num_threads(num_thread);

    start_time = omp_get_wtime();

    // switch case for scheduling type 1~4
    switch (scheduling_type)
    {
    case 1:
#pragma omp parallel for reduction(+ : sum) default(shared) private(i) schedule(static)
        for (i = 0; i < num_end; i++)
        {
            if (is_prime(i))
            {
                sum++;
            }
        }
        break;

    case 2:
#pragma omp parallel for reduction(+ : sum) default(shared) private(i) schedule(dynamic)
        for (i = 0; i < num_end; i++)
        {
            if (is_prime(i))
            {
                sum++;
            }
        }
        break;

    case 3:
#pragma omp parallel for reduction(+ : sum) default(shared) private(i) schedule(static, 10)
        for (i = 0; i < num_end; i++)
        {
            if (is_prime(i))
            {
                sum++;
            }
        }
        break;

    case 4:
#pragma omp parallel for reduction(+ : sum) default(shared) private(i) schedule(dynamic, 10)
        for (i = 0; i < num_end; i++)
        {
            if (is_prime(i))
            {
                sum++;
            }
        }
        break;
    }

    end_time = omp_get_wtime();
    double timeDiff = (end_time - start_time) * 1000;
    printf("Execution Time : %lfms\n", timeDiff);

    printf("1...%d prime# counter= %d\n", num_end - 1, sum);

    return 0;
}