all: parallel_random_forest

seq_random_forest: seq_random_forest.c
	gcc -Wall -o seq_random_forest seq_random_forest.c cycletimer.c

parallel_random_forest: parallel_random_forest.c
	gcc -Wall -fopmenmp -o parallel_random_forest parallel_random_forest.c cycletimer.c

clean:
	rm seq_random_forest
	rm parallel_random_forest
