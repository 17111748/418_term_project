all: baseline_parallel seq_random_forest level_parallel

seq_random_forest: seq_random_forest.c
	gcc -Wall -o seq_random_forest seq_random_forest.c cycletimer.c

baseline_parallel: baseline_parallel.c
	gcc -Wall -fopenmp -o baseline_parallel baseline_parallel.c cycletimer.c

level_parallel: level_parallel.c
	gcc -Wall -fopenmp -o level_parallel level_parallel.c cycletimer.c

clean:
	rm seq_random_forest
	rm baseline_parallel
	rm level_parallel
