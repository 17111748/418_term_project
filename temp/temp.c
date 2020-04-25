#include <stdio.h>
#include <malloc.h>
#include <stdbool.h>
#include <limits.h>
#include <stdlib.h>

#include "cycletimer.h"
//#include "cycletimer.c"

int main()
{
    int i, j, y;
    double startSeconds = currentSeconds();
    int* temp = malloc(sizeof(int) * 1000); 

    // clock_t start = clock(), diff; 
    // ProcessIntenseFunction(); 
    //#pragma omp parallel for
    for (i = 0; i < 100000000; i++){
        for (j = 0; j < 1000; j++) {
            temp[j] = 2 * i; 
        }
    }

    y = temp[555]; 
    double endSeconds = currentSeconds();
    // diff = clock() - start; 
    // int msec = diff * 1000 / CLOCKS_PER_SEC; 

    // printf("Time taken %d seconds %d milliseconds", msec/1000, msec % 1000); 
    printf("Time: %f\n", endSeconds - startSeconds);
    printf("Y: %d\n", y);
    return 0;
}
