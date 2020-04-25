#include <stdio.h>
#include <malloc.h>
#include <stdbool.h>
#include <limits.h>
#include <stdlib.h>

#include "cycletimer.h"
//#include "cycletimer.c"

int main()
{
    long i; 

    long y = 0; 
    int z = 0; 
    double startSeconds = currentSeconds();
    // clock_t start = clock(), diff; 
    // ProcessIntenseFunction(); 
    #pragma omp parallel for
    for (i = 0; i < 100000000; i++){
        y = 2 * i;
    }

    double endSeconds = currentSeconds();
    #pragma omp barrier 
    // diff = clock() - start; 
    // int msec = diff * 1000 / CLOCKS_PER_SEC; 

    // printf("Time taken %d seconds %d milliseconds", msec/1000, msec % 1000); 
    printf("Start Time: %f\n", startSeconds);
    printf("End Time: %f\n", endSeconds);
    printf("Time: %f\n", endSeconds - startSeconds);
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
