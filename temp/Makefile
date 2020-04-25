CC=gcc
CFLAGS=-g -O3 -Wall -DDEBUG=0
OMP=-fopenmp -DOMP
CFILES = cycletimer.c
HFILES = cycletimer.h



all: temp_parallel

temp: temp.c
	$(CC) $(CFLAGS) -o temp temp.c cycletimer.c

temp_parallel: temp_parallel.c
	$(CC) $(CFLAGS) $(OMP) -o temp_parallel temp_parallel.c cycletimer.c



clean:
	rm temp
