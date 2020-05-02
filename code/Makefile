all: seq_random_forest baseline_parallel level_parallel node_parallel_1 node_parallel_2 node_parallel_3 node_parallel_4

seq_random_forest: seq_random_forest.c
	gcc -Wall -o seq_random_forest seq_random_forest.c cycletimer.c

baseline_parallel: baseline_parallel.c
	gcc -Wall -fopenmp -o baseline_parallel baseline_parallel.c cycletimer.c

level_parallel: level_parallel.c
	gcc -Wall -fopenmp -o level_parallel level_parallel.c cycletimer.c

node_parallel_1: node_parallel_1.c
	gcc -Wall -fopenmp -o node_parallel_1 node_parallel_1.c cycletimer.c

node_parallel_2: node_parallel_2.c
	gcc -Wall -fopenmp -o node_parallel_2 node_parallel_2.c cycletimer.c

node_parallel_3: node_parallel_3.c
	gcc -Wall -fopenmp -o node_parallel_3 node_parallel_3.c cycletimer.c

node_parallel_4: node_parallel_4.c
	gcc -Wall -fopenmp -o node_parallel_4 node_parallel_4.c cycletimer.c

albert_level_parallel: albert_level_parallel.c
	gcc -Wall -fopenmp -o albert_level_parallel albert_level_parallel.c cycletimer.c

test_parallel: test_parallel.c
	gcc -Wall -fopenmp -o test_parallel test_parallel.c cycletimer.c

clean:
	rm seq_random_forest
	rm baseline_parallel
	rm level_parallel
	rm node_parallel_1
	rm node_parallel_2
	rm node_parallel_3
	rm node_parallel_4
