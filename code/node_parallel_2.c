/*  Sequential Version of the Random Forest Algorithm */ 

#include <stdio.h>
#include <malloc.h>
#include <stdbool.h>
#include <limits.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <getopt.h>
#include <omp.h>
#include "cycletimer.h"

// int NUM_ENTRIES_TOTAL = 569;
// int NUM_TEST_ENTRIES = 85; // Approx 15%
// int NUM_TRAIN_ENTRIES = 484; 
int NUM_FEATURES = 30;
// int NUM_FEATURES = 14; 

int STACK_CAPACITY = 10000; 
int QUEUE_CAPACITY = 10000; 

typedef struct dataidxs {
	int *data_idxs;
	int n_entries;
} dataidxs_t; 


typedef struct group {
	dataidxs_t *left_idxs; 
	dataidxs_t *right_idxs;
} group_t; 

typedef struct node {
	struct node *left; 
	struct node *right; 
	float feature_value;
	group_t *group; 
	int feature;
	int depth;
	int result;
	bool leaf;   
} node_t; 

typedef struct tree {
	node_t *root_node; 
} tree_t; 

typedef struct stack {
	int end;  
	int capacity; 
	node_t **nodeList; 
} stack_t; 

typedef struct queue {
	int start; 
	int end; 
	int capacity; 
	node_t **nodeList; 
} queue_t; 

bool float_equal(float a, float b)
{
	float epsilon = 0.00001;
 	return fabs(a - b) < epsilon;
}

// stack_t *create_stack() {
// 	stack_t *stack = malloc(sizeof(stack_t)); 
// 	stack->end = 0; 
// 	stack->capacity = STACK_CAPACITY; 
// 	stack->nodeList = malloc(sizeof(node_t*) * stack->capacity); 
// 	return stack; 
// }

// bool stack_isEmpty(stack_t *stack) {
// 	return stack->end == 0; 
// }

// void stack_push(stack_t *stack, node_t *node) {
// 	if (stack->end == stack->capacity) {
// 		printf("Failed to Push onto Stack \n"); 
// 		return; 
// 	}
// 	stack->nodeList[stack->end] = node; 
// 	stack->end = stack->end + 1; 
// }

// node_t *stack_pop(stack_t *stack) {
// 	if (stack->end == 0) return NULL; 
// 	stack->end = stack->end - 1; 
// 	return stack->nodeList[stack->end + 1]; 
// }

// queue_t *create_queue() {
// 	queue_t *queue = malloc(sizeof(queue_t)); 
// 	queue->start = 0; 
// 	queue->end = 0; 
// 	queue->capacity = QUEUE_CAPACITY; 
// 	queue->nodeList = malloc(sizeof(node_t*) * queue->capacity); 
// 	return queue; 
// }

// bool queue_isEmpty(queue_t *queue) {
// 	return (queue->start == queue->end) && !(((queue->end + 1) % queue->capacity) == queue->start); 
// }

// void queue_push(queue_t *queue, node_t *node) {
// 	if(((queue->end + 1) % queue->capacity) == queue->start) {
// 		printf("Failed to Push onto Queue \n"); 
// 		return; 
// 	}
// 	queue->nodeList[queue->end] = node; 
// 	queue->end = (queue->end + 1) % queue->capacity; 
// }

// node_t *queue_pop(queue_t *queue) {
// 	if (queue->end == queue->start) return NULL; 
// 	node_t *node = queue->nodeList[queue->start]; 
// 	queue->start = (queue->start + 1) % queue->capacity; 
// 	return node; 
// }


node_t* create_node(int feature, float feature_value, group_t *group, int depth) {
	node_t *node = malloc(sizeof(node_t)); 
	node->left = NULL; 
	node->right = NULL; 
	node->feature = feature; 
	node->group = group; 
	node->depth = depth;
	node->feature_value = feature_value; 
	node->result = -1; 
	node->leaf = false; 
	return node;
}

dataidxs_t* create_dataidxs(int n_entries) {
	dataidxs_t *dataidxs = malloc(sizeof(dataidxs_t)); 
	int *idxs = malloc(sizeof(int) * n_entries); 
	dataidxs->data_idxs = idxs;
	dataidxs->n_entries = n_entries; 
	return dataidxs;
}


// Tested 
/* Create a random subsample from the dataset with replacement */
dataidxs_t* subsample(int n_entries, float percentage) {
	int i;
	int n_sample = (float)(n_entries * percentage); 
	dataidxs_t *sample = create_dataidxs(n_sample);
	// for (i = 0; i < n_sample; i++) {
	// 	int index = rand() % n_entries; 
	// 	sample->data_idxs[i] = index; 
	// }
	for (i = 0; i < n_sample; i++) {
		// int index = rand() % n_entries; 
		sample->data_idxs[i] = i; 
	}
	return sample; 
}


// Tested 
float gini_index(float **train_set, group_t *group, int n_features) {
	dataidxs_t *left_idxs = group->left_idxs; 
	dataidxs_t *right_idxs = group->right_idxs; 

	int n_instances = left_idxs->n_entries + right_idxs->n_entries; 


	float gini = 0.0; 
	int k; 
	float size, score, p0, p1; 
	

	// Left Side 
	size = (float)(left_idxs->n_entries); 
	if (size != 0) {
		score = 0.0; 
		p0 = 0.0; 
		p1 = 0.0; 
		for (k = 0; k < left_idxs->n_entries; k++) {
			int index = left_idxs->data_idxs[k]; 
			// get malign or not count 
			if (float_equal(train_set[index][n_features], 0.0)) {
				p0 += 1; 
			}
			else {
				p1 += 1; 
			}
		}
		score += p0/size * p0/size;
		score += p1/size * p1/size; 
		gini += (1.0 - score) * (size / (float)n_instances); 
	}

	// Right Side 
	size = (float)(right_idxs->n_entries); 
	if (size != 0) {
		score = 0.0; 
		p0 = 0.0; 
		p1 = 0.0; 
		for (k = 0; k < right_idxs->n_entries; k++) {
			int index = right_idxs->data_idxs[k]; 
			if (float_equal(train_set[index][n_features], 0.0)) {
				p0 += 1;
			}
			else {
				p1 += 1;
			}
		}
		score += p0/size * p0/size;
		score += p1/size * p1/size; 
		gini += (1.0 - score) * (size / (float)n_instances); 
	}
	
	return gini; 
}



void free_group(group_t *group) {
	free(group->left_idxs->data_idxs);
	free(group->left_idxs);
	free(group->right_idxs->data_idxs);
	free(group->right_idxs);
	free(group);
}

void free_tree_groups(node_t *node) {
	if (node->leaf) {
		return;
	}
	free_group(node->group);
	if (node->left) {
		free_tree_groups(node->left);
	}
	else if (node->right) {
		free_tree_groups(node->right);
	}
}

void free_tree(node_t *node) {
	if (node->leaf) {
		free(node);
		return;
	}
	if (node->left) {
		free_tree(node->left);
	}
	else if (node->right) {
		free_tree(node->right);
	}
	free(node);
}

// Tested 
group_t *test_split(int index, float value, float **train_set, dataidxs_t *dataset) {
	dataidxs_t *left = create_dataidxs(dataset->n_entries); 
	dataidxs_t *right = create_dataidxs(dataset->n_entries); 

	int left_count = 0;
	int right_count = 0; 

	int i; 
	for (i = 0; i < dataset->n_entries; i++) {
		int row = dataset->data_idxs[i]; 
		if (train_set[row][index] < value) {
			left->data_idxs[left_count] = row; 
			left_count++; 
		}
		else {
			right->data_idxs[right_count] = row; 
			right_count++; 
		}
	} 

	left->n_entries = left_count; 
	right->n_entries = right_count; 

	group_t *group = malloc(sizeof(group_t)); 
	group->left_idxs = left; 
	group->right_idxs = right; 

	return group; 
}

void print_node_t(node_t *n) {
	printf("\n~~~~~~~~ Print Node_t ~~~~~~~~~~~~~\n"); 
	// node_t *left = n->left; 
	// node_t *right = n->right;
	printf("Depth: %d\n\n", n->depth); 
	
	printf("Feature Index: %d\n", n->feature);
	printf("Feature Value: %f\n\n", n->feature_value); 
	
	printf("Result: %d\n", n->result); 
	printf("Leaf: %d\n", (n->leaf == true)); 
	
	
	printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"); 
}

// node_t *get_split(float **train_set, dataidxs_t *dataset, int n_features, int node_depth) {
// 	int best_feature_index = -1; 
// 	float best_feature_value = -1; 
// 	float best_score = (float)INT_MAX; 
// 	group_t *best_group = NULL;

// 	int index; 
// 	int count; 
// 	int i, indexD;
// 	float gini;  

// 	// Randomly Select N features from featureList 
// 	int featureList[n_features]; 
// 	featureList[0] = rand() % n_features; 
// 	for (i = 1; i < n_features; i++) {
// 		count = 0; 
// 		index = rand() % n_features; 
// 		while (count < i) {
// 			if(featureList[count] == index) {
// 				index = rand() % n_features; 
// 				count = 0; 
// 			}
// 			else {
// 				count++; 
// 			}
// 		}
// 		featureList[i] = index; 
// 	}
	
// 	// for (i = 0; i < n_features; i++) {
// 	// 	featureList[i] = i;
// 	// }

// 	// Selecting the best split with the lowest gini index

// 	// #pragma omp parallel
// 	for (index = 0; index < n_features; index++) {
// 		for (indexD = 0; indexD < dataset->n_entries; indexD++) {
// 			int feature_index = featureList[index]; 
// 			int data_index = (dataset->data_idxs)[indexD]; 

// 			group_t *group = test_split(feature_index, train_set[data_index][feature_index], train_set, dataset); 
// 			gini = gini_index(train_set, group, n_features); 
// 			if (gini < best_score) {
// 				best_feature_index = feature_index; 
// 				best_feature_value = train_set[data_index][feature_index]; 
// 				best_score = gini; 
// 				best_group = group; 
// 			}
// 			else {
// 				free_group(group);
// 			}
// 		}
// 	}

// 	node_t *node = create_node(best_feature_index, best_feature_value, best_group, node_depth);

// 	// print_node_t(node);  

// 	return node;
// }

node_t *get_split(float **train_set, dataidxs_t *dataset, int n_features, int node_depth) {
	
	int i; 
	int count; 
	int index_i; 

	// Randomly Select N features from featureList 
	int featureList[n_features]; 
	featureList[0] = rand() % n_features; 
	for (i = 1; i < n_features; i++) {
		count = 0; 
		index_i = rand() % n_features; 
		while (count < i) {
			if(featureList[count] == index_i) {
				index_i = rand() % n_features; 
				count = 0; 
			}
			else {
				count++; 
			}
		}
		featureList[i] = index_i; 
	}
	
	for (i = 0; i < n_features; i++) {
		featureList[i] = i;
	}



	int num_threads = omp_get_max_threads(); 

	float best_scores[num_threads]; 
	int best_feature_indexs[num_threads]; 
	float best_feature_values[num_threads]; 
	group_t *best_groups[num_threads];  


	// Selecting the best split with the lowest gini index
	int index; 
	int indexD;
	#pragma omp parallel 
	{
		int tid = omp_get_thread_num(); 
		best_feature_indexs[tid] = -1; 
		best_feature_values[tid] = -1; 
		best_scores[tid] = (float)INT_MAX; 
		best_groups[tid] = NULL;
		#pragma omp for schedule(static) collapse(2)
		for (index = 0; index < n_features; index++) { 
			for (indexD = 0; indexD < dataset->n_entries; indexD++) {
				int feature_index = featureList[index]; 
				int data_index = (dataset->data_idxs)[indexD]; 
				group_t *group = test_split(feature_index, train_set[data_index][feature_index], train_set, dataset); 
				float gini = gini_index(train_set, group, n_features); 
				if (gini < best_scores[tid]) {
					best_feature_indexs[tid] = feature_index; 
					best_feature_values[tid] = train_set[data_index][feature_index]; 
					best_scores[tid] = gini; 
					best_groups[tid] = group; 
				}
				else {
					free_group(group);
				}
			}
		}
	}

	float best_score = (float)INT_MAX; 
	int best_feature_index = -1; 
	float best_feature_value = -1; 
	group_t *best_group = NULL; 

	for (i = 0; i < num_threads; i++) {
		if (best_scores[i] < best_score) {
			best_score = best_scores[i]; 
			best_feature_index = best_feature_indexs[i]; 
			best_feature_value = best_feature_values[i]; 
			best_group = best_groups[i]; 
		}
	}
	
	node_t *node = create_node(best_feature_index, best_feature_value, best_group, node_depth); 

	// print_node_t(node); 

	return node;
}


node_t *create_leaf(float **train_set, dataidxs_t *dataset, int node_depth, int n_features) {
	int yes_count = 0; 
	int no_count = 0; 
	int i; 
	for (i = 0; i < dataset->n_entries; i++) {
		int index = dataset->data_idxs[i]; 
		if (float_equal(train_set[index][n_features], 1.0)) {
			yes_count++; 
		}
		else if (float_equal(train_set[index][n_features], 0.0)) {
			no_count++; 
		}
		else {
			printf("Create Leaf: Should not get here \n"); 
		}
	}

	node_t* node = create_node(-1, -1, NULL, node_depth); 
	node->leaf = true; 
	node->result = (yes_count >= no_count) ? 1 : 0; 

	return node; 
}


void split(node_t *node, int max_depth, int min_size, int n_features, float **train_set) {
	double startSeq, endSeq; 
	int i;
	node_t *cur_node;
	group_t *group;
	dataidxs_t *left; 
	dataidxs_t *right;
	int max_capacity = 10000;
	node_t *temp_node;
	node_t **temp;
	node_t **cur_level = malloc(sizeof(node_t*) * max_capacity);
	node_t **next_level = malloc(sizeof(node_t*) * max_capacity);
	int cur_level_count = 0;
	int next_level_count = 0;

	cur_level[cur_level_count] = node;
	cur_level_count++;

	while (cur_level_count > 0) {
		// printf("CUR_LEVEL_COUNT: %d\n", cur_level_count);
		startSeq = currentSeconds(); 

		// for(i = 0; i < cur_level_count; i++) {
		// 	printf("i: %d ", i); 
		// 	print_node_t(cur_level[i]); 

		// }
		// if (cur_level_count == 2) exit(1); 
		for (i = 0; i < cur_level_count; i++) {
			cur_node = cur_level[i];
			group = cur_node->group; 
			left = group->left_idxs; 
			right = group->right_idxs;
			if (left->n_entries == 0 || right->n_entries == 0) {
				if (left->n_entries == 0) {
					temp_node = create_leaf(train_set, right, cur_node->depth, n_features);
				}
				else {
				// else if (right->n_entries == 0) {
					temp_node = create_leaf(train_set, left, cur_node->depth, n_features);
				}
				cur_node->leaf = true;
				cur_node->result = temp_node->result;
				free_tree_groups(cur_node);
				free_tree_groups(temp_node);
				free(temp_node);
				continue;
			}

			if (cur_node->depth >= max_depth - 1) {
				cur_node->left = create_leaf(train_set, left, cur_node->depth + 1, n_features);
				cur_node->right = create_leaf(train_set, right, cur_node->depth + 1, n_features);
				continue;
			}

			if (left->n_entries <= min_size) {
				cur_node->left = create_leaf(train_set, left, cur_node->depth + 1, n_features);
			}
			else {
				cur_node->left = get_split(train_set, left, n_features, cur_node->depth + 1);
				next_level[next_level_count] = cur_node->left;
				next_level_count++;
				if (next_level_count >= max_capacity) printf("\n\nERROR IN SPLIT: MAX CAPACITY REACHED\n\n");
			}
			if (right->n_entries <= min_size) {
				cur_node->right = create_leaf(train_set, right, cur_node->depth + 1, n_features);
			}
			else {
				cur_node->right = get_split(train_set, right, n_features, cur_node->depth + 1);
				next_level[next_level_count] = cur_node->right;
				next_level_count++;
				if (next_level_count >= max_capacity) printf("\n\nERROR IN SPLIT: MAX CAPACITY REACHED\n\n");
			}
		}

		// endSeq = currentSeconds(); 
		// printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"); 
		// printf("seq (split) %f\n", endSeq - startSeq); 
		// printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");

		temp = cur_level;
		cur_level_count = next_level_count;
		cur_level = next_level;
		next_level_count = 0;
		next_level = temp;
	}

	free(cur_level);
	free(next_level);
}


node_t *build_tree(float **train_set, dataidxs_t *sample, int max_depth, int min_size, 
					int n_features) {
	node_t *root_node = get_split(train_set, sample, n_features, 0);
	split(root_node, max_depth, min_size, n_features, train_set); 
	return root_node; 
}


// Pass in n_trees
int predict(tree_t *tree, float **test_set, int row) {
	node_t *cur_node = tree->root_node; 
	float *test_row = test_set[row]; 
	
	int feature; 
	float feature_value; 
	if (cur_node == NULL) return -1; 
	// while ((cur_node->left == NULL) && (cur_node->right == NULL)) {
	while(cur_node->leaf != true) {
		feature = cur_node->feature; 
		feature_value = cur_node->feature_value; 
		if (test_row[feature] < feature_value) {
			cur_node = cur_node->left; 
		}
		else {
			cur_node = cur_node->right; 
		}
	}

	return cur_node->result; 
}

// TODO: Find maximum occurence 
int bagging_predict(tree_t **tree_list, int n_trees, float **test_set, int row) {
	int i; 
	int prediction; 
	int predict_0 = 0;  
	int predict_1 = 0;  

	// double startPredict, endPredict; 
	for (i = 0; i < n_trees; i++) {
		// if (i == 0) {
		// 	startPredict = currentSeconds(); 
		// }
		prediction = predict(tree_list[i], test_set, row); 
		// if (i == 0) {
		// 	endPredict = currentSeconds(); 
		// 	printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"); 
		// 	printf("Time to Predict From One Tree (bagging_predict) %f\n", endPredict - startPredict); 
		// 	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
		// }
		if (prediction == 0) predict_0++; 
		else if (prediction == 1) predict_1++; 
	}

	return (predict_0 > predict_1) ? 0 : 1; 
}


// Dataset should be int **
float random_forest(float **train_set, float **test_set, int train_len, int test_len,
				   int max_depth, int min_size, float percentage,
				   int n_trees, int n_features) {

	double startSingleTree; 
	double endSingleTree; 

	tree_t **tree_list = malloc(n_trees * sizeof(tree_t*)); 

	int tree_index;

	double startSeconds = currentSeconds();

	// #pragma omp parallel for //schedule(dynamic)
	for (tree_index = 0; tree_index < n_trees; tree_index++) {
		dataidxs_t *sample = subsample(train_len, percentage); 
		if (tree_index == 0) {
			startSingleTree = currentSeconds(); 
		}
		node_t *root_node = build_tree(train_set, sample, max_depth, min_size, n_features); 
		if (tree_index == 0) {
			endSingleTree = currentSeconds(); 
			printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"); 
			printf("Time for a single tree (build_tree) %f\n", endSingleTree - startSingleTree); 
			printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");
		}

		tree_t *tree = malloc(sizeof(tree_t)); 
		tree->root_node = root_node; 
		tree_list[tree_index] = tree; 

		free(sample->data_idxs);
		free(sample);

		free_tree_groups(root_node);
	}

	double endSeconds = currentSeconds(); 
	printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"); 
	printf("Time to Build Forest - %d trees (random_forest) %f\n", n_trees, endSeconds - startSeconds); 
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");

	double startPredict = currentSeconds();
	double startBagging, endBagging; 

	int row, prediction;
	int correct = 0;
	int incorrect = 0;
	// int *predictions = malloc(test_len * sizeof(int)); 
	for (row = 0; row < test_len; row++) {
		if (row == 0) {
			startBagging = currentSeconds(); 
		}
		prediction = bagging_predict(tree_list, n_trees, test_set, row);
		if (row == 0) {
			endBagging = currentSeconds(); 
			printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"); 
			printf("Time to Predict One Row (random_forest) %f\n", endBagging - startBagging); 
			printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");
		}
		if (float_equal((float) prediction, test_set[row][n_features])) {
			correct++;
		}
		else {
			incorrect++;
		}
	}

	double endPredict = currentSeconds(); 
	printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"); 
	printf("Time to Predict All Rows (random_forest) %f\n", endPredict - startPredict); 
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");

	for (tree_index = 0; tree_index < n_trees; tree_index++) {
		free_tree(tree_list[tree_index]->root_node);
		free(tree_list[tree_index]);
	}
	free(tree_list);

	return ((float) correct) / ((float) test_len) * 100.0;
}


float* get_row(char* line, int num)
{
    const char* tok = strtok(line, ",");
    float *arr = malloc(sizeof(float) * 40);
    int i;
    for (i = 0; i < num; i++) {
    	arr[i] = atof(tok);
    	tok = strtok(NULL, ",");
    }
    return arr;
}

int main(int argc, char **argv)
{
	extern char *optarg; 

	int num_threads = 1; 
	int file_size = 0; 

	int opt;

	while ((opt = getopt(argc, argv, "t:f:")) != -1) {
		switch (opt) {
			case 't': 
				num_threads = atoi(optarg);  
				break;

			case 'f': 
				file_size = atoi(optarg); 
				break;  

			default: 
				fprintf(stderr, "Error- Invalid opt: %d\n", opt); 
				exit(1); 
		}
	}

	omp_set_num_threads(num_threads);
	// omp_set_num_threads(8);


	// Testing omp_get_max and omp_get_num

	// printf("Get Max Num Threads: %d\n", omp_get_max_threads());
	// printf("Get Num Threads: %d\n", omp_get_num_threads());

	// #pragma omp parallel
	// {
	// 	printf("Inside Get Max Num Threads: %d\n", omp_get_max_threads());
	// 	printf("Inside Get Num Threads: %d\n", omp_get_num_threads());

	// 	#pragma omp barrier 

	// 	omp_set_num_threads(4); 

	// 	int i; 

	// 	printf("\n\n"); 
	// 	#pragma omp for 
	// 	for(i = 0; i < 8; i++) {
	// 		printf("Second Get Max Num Threads: %d\n", omp_get_max_threads());
	// 		printf("Second Get Num Threads: %d\n", omp_get_num_threads());
	// 	}

	// }


    FILE* stream;
	switch (file_size) {
		case 0: //xsmall
    		stream = fopen("../data/clean_data.csv", "r");
    		break;
		case 1: //small
    		stream = fopen("../data/x5_clean_data.csv", "r");
    		break;
    	case 2: //medium
    		stream = fopen("../data/medium_clean_data.csv", "r");
    		break;
    	case 3: //large
    		stream = fopen("../data/large_clean_data.csv", "r");
    		break;
    	case 4: // random, n=10
    		stream = fopen("../data/random_data_n_10.csv", "r");
    		break;
		case 5: // random, n=100
			stream = fopen("../data/random_data_n_100.csv", "r");
			break;
		case 6: // random, n=1000
    		stream = fopen("../data/random_data_n_1000.csv", "r");
    		break;
    	case 7: // random, n=10000
    		stream = fopen("../data/random_data_n_10000.csv", "r");
    		break;
    	case 8: // random, n=20000
    		stream = fopen("../data/random_data_n_20000.csv", "r");
    		break;
    	default:
    		stream = fopen("../data/clean_data.csv", "r");
    		break;
	}

	double startSeconds = currentSeconds();

    float **data = malloc(sizeof(float *) * 100000);
    char line[4096];
    int count = 0;
    while (fgets(line, 4096, stream))
    {
        char* tmp = strdup(line);
       	float *arr = get_row(tmp, NUM_FEATURES+1);
       	data[count] = arr;
       	count++;
        free(tmp);
    }

    int n_train_entries = (int)(0.8*(float)count);

    double endSeconds = currentSeconds(); 
	printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"); 
	printf("Time to read and initialize data: %f\n", endSeconds - startSeconds); 
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");

    float accuracy = random_forest(
    					&data[0],						// train set
    					&data[n_train_entries],			// test set
    					n_train_entries,				// n_train_entries
    					count - n_train_entries,		// n_test_entries
    					20, 							// max depth
    					2,								// min size
    					1.0,							// ratio
    					1,								// n_trees
    					NUM_FEATURES);					// n_features (no. cols in dataset - 1)


    printf("accuracy: %f\n", accuracy);

    int i;

    for (i = 0; i < count; i++) {
    	free(data[i]);
    }
    free(data);

    return 0; 
}
