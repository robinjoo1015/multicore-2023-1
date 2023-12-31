#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void *PrintHello(void *ptr)
{
   char *filename;
   int j;
   filename = (char *) ptr;
   while (1) {
      printf("Hello World! It's me, thread %s!\n", filename);
      sleep(1);
   }
   pthread_exit(NULL);
}

int main (int argc, char *argv[])
{
   pthread_t thread[100];
   int err_code, i=0;
   char *filename;

   printf ("Enter thread name at any time to create thread\n");

   while (1) {
      filename = (char *) malloc (80*sizeof(char));
      scanf ("%s", filename);
      printf("In main: creating thread %d\n", i);
      err_code = pthread_create(&thread[i], NULL, PrintHello, (void *)filename);
      if (err_code){
         printf("ERROR code is %d\n", err_code);
         exit(-1);
      } else i++;
   }
   pthread_exit(NULL);
}

