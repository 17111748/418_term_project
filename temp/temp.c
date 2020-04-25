#include <stdio.h>
#include <malloc.h>
#include <stdbool.h>
#include <limits.h>
#include <stdlib.h>

#include "cycletimer.h"
//#include "cycletimer.c"

int main()
{
    long i, y;
    double startSeconds = currentSeconds();
    // clock_t start = clock(), diff; 
    // ProcessIntenseFunction(); 
    //#pragma omp parallel for
    for (i = 0; i < 100000000; i++){
        y = 2 * i;
    }
    double endSeconds = currentSeconds();
    // diff = clock() - start; 
    // int msec = diff * 1000 / CLOCKS_PER_SEC; 

    // printf("Time taken %d seconds %d milliseconds", msec/1000, msec % 1000); 
    printf("Time: %f\n", endSeconds - startSeconds);
    printf("Y: %d\n", y);
    return 0;
}
