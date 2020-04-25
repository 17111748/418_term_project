#include <stdio.h>
#include <malloc.h>
#include <stdbool.h>
#include <limits.h>
#include <stdlib.h>
#include <omp.h>

#include "cycletimer.h"
//#include "cycletimer.c"


int extra(int j, int *temp) {
    int i = 0; 
    while(i < j) {
        temp[i] = temp[i] + 1; 
        i++; 
    }

    return 1; 
}


int main()
{
    int thread_count = 8; 
    omp_set_num_threads(thread_count);
    int i, j, y;

    int num2 = 1000000000; 
    int *temp2 = malloc(sizeof(int) * num2); 
    int *temp3 = malloc(sizeof(int) * num2); 
    for(i = 0; i < num2; i++) {
        temp2[i] = 133 * i + 123; 
        temp3[i] = 0; 
    }

    int num = num2; 
    double startSeconds = currentSeconds();

    // clock_t begin = clock();

    int* temp = malloc(sizeof(int) * num); 

    
    #pragma omp parallel for schedule(static, 128) 
    for (j = 0; j < num; j++) {
        // temp[j] = temp2[j] + extra(j, temp3);
        y = j * 2; 

    }
    

    // y = temp[111]; 
    double endSeconds = currentSeconds();

    // clock_t end = clock();
    // double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    // diff = clock() - start; 
    // int msec = diff * 1000 / CLOCKS_PER_SEC; 

    // printf("Time taken %d seconds %d milliseconds", msec/1000, msec % 1000); 
    printf("Start Time: %f\n", startSeconds);
    printf("End Time: %f\n", endSeconds);
    printf("Time: %f\n", endSeconds - startSeconds);
    // printf("Time: %f\n", time_spent);
    printf("Y: %d\n\n", y);



    // y = 0; 
    // double startSeconds1 = currentSeconds();
    // // clock_t start = clock(), diff; 
    // // ProcessIntenseFunction(); 
    // for (i = 0; i < 100000000; i++){
    //     z = 3 * i;
    // }


    // double endSeconds1 = currentSeconds();

    // printf("Start Time: %f\n", startSeconds1);
    // printf("End Time: %f\n", endSeconds1);
    // printf("Time: %f\n", endSeconds1 - startSeconds1);
    // printf("Y: %d\n", z);


    return 0;
}

// /******************************************************************************
// * FILE: omp_hello.c
// * DESCRIPTION:
// *   OpenMP Example - Hello World - C/C++ Version
// *   In this simple example, the master thread forks a parallel region.
// *   All threads in the team obtain their unique thread number and print it.
// *   The master thread only prints the total number of threads.  Two OpenMP
// *   library routines are used to obtain the number of threads and each
// *   thread's number.
// * AUTHOR: Blaise Barney  5/99
// * LAST REVISED: 04/06/05
// ******************************************************************************/
// #include <omp.h>
// #include <stdio.h>
// #include <stdlib.h>

// int main (int argc, char *argv[]) 
// {
//     int nthreads, tid;

//     /* Fork a team of threads giving them their own copies of variables */
//     #pragma omp parallel private(nthreads, tid)
//     {
//         /* Obtain thread number */
//         tid = omp_get_thread_num();
//         printf("Hello World from thread = %d\n", tid);

//       /* Only master thread does this */
//         if (tid == 0) 
//         {
//             nthreads = omp_get_num_threads();
//             printf("Number of threads = %d\n", nthreads);
//         }

//     }  /* All threads join master thread and disband */

// }
