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

int NUM_FEATURES = 30;

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

dataidxs_t* subsample(int n_entries, float percentage) {
	int i;
	int n_sample = (float)(n_entries * percentage); 
	dataidxs_t *sample = create_dataidxs(n_sample);
	for (i = 0; i < n_sample; i++) {
		int index = rand() % n_entries; 
		sample->data_idxs[i] = index; 
	}
	return sample; 
}

float gini_index(float **train_set, group_t *group, int n_features) {
	dataidxs_t *left_idxs = group->left_idxs; 
	dataidxs_t *right_idxs = group->right_idxs; 

	int n_instances = left_idxs->n_entries + right_idxs->n_entries; 


	float gini = 0.0; 
	int k; 
	float size, score, p0, p1; 
	
	size = (float)(left_idxs->n_entries); 
	if (size != 0) {
		score = 0.0; 
		p0 = 0.0; 
		p1 = 0.0; 
		for (k = 0; k < left_idxs->n_entries; k++) {
			int index = left_idxs->data_idxs[k];
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

node_t *get_split(float **train_set, dataidxs_t *dataset, int n_features, int node_depth) {
	int best_feature_index = -1; 
	float best_feature_value = -1; 
	float best_score = (float)INT_MAX; 
	group_t *best_group = NULL;

	int index; 
	int count; 
	int i, indexD;
	float gini;  

	// Randomly Select N features from featureList 
	int featureList[n_features]; 
	featureList[0] = rand() % n_features; 
	for (i = 1; i < n_features; i++) {
		count = 0; 
		index = rand() % n_features; 
		while (count < i) {
			if(featureList[count] == index) {
				index = rand() % n_features; 
				count = 0; 
			}
			else {
				count++; 
			}
		}
		featureList[i] = index; 
	}
	
	// for (i = 0; i < n_features; i++) {
	// 	featureList[i] = i;
	// }

	// Selecting the best split with the lowest gini index

	for (index = 0; index < n_features; index++) {
		for (indexD = 0; indexD < dataset->n_entries; indexD++) {
			int feature_index = featureList[index]; 
			int data_index = (dataset->data_idxs)[indexD]; 

			group_t *group = test_split(feature_index, train_set[data_index][feature_index], train_set, dataset); 
			gini = gini_index(train_set, group, n_features); 
			if (gini < best_score) {
				best_feature_index = feature_index; 
				best_feature_value = train_set[data_index][feature_index]; 
				best_score = gini; 
				best_group = group; 
			}
			else {
				free_group(group);
			}
		}
	}

	node_t *node = create_node(best_feature_index, best_feature_value, best_group, node_depth); 

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


int weight(node_t *node) {
    return node->group->left_idxs->n_entries + node->group->right_idxs->n_entries;
}

// A utility function to swap two elements 
void swap(node_t** a, node_t** b) 
{
    node_t *temp = *a; 
    *a = *b;
    *b = temp;
}

int partition(node_t *arr[], int low, int high)
{
    int pivot = weight(arr[high]);    // pivot 
    int i = (low - 1);  // Index of smaller element 
  	int j;
    for (j = low; j <= high- 1; j++) 
    {
        if (weight(arr[j]) < pivot) 
        { 
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1); 
}
  
/* The main function that implements QuickSort 
   arr[] --> Array to be sorted, 
   low  --> Starting index (inclusive), 
   high  --> Ending index (inclusive) */
void quickSort(node_t *arr[], int low, int high) 
{ 
    if (low < high) 
    {
        int pi = partition(arr, low, high); 
        quickSort(arr, low, pi - 1); 
        quickSort(arr, pi + 1, high); 
    }
}

void split(node_t *node, int max_depth, int min_size, int n_features, float **train_set) {
	int i, j, min_thread_idx, min_weight;
	int max_capacity = 10000;
	node_t **temp;
	node_t **cur_level = malloc(sizeof(node_t*) * max_capacity);
	node_t **next_level = malloc(sizeof(node_t*) * max_capacity);
	int cur_level_count = 0;
	int next_level_count = 0;
	int cur_weights[omp_get_max_threads()];
	int cur_counts[omp_get_max_threads()];
	int *cur_idxs[omp_get_max_threads()];

	for (i = 0; i < omp_get_max_threads(); i++) {
		cur_idxs[i] = malloc(1000 * sizeof(int));
	}

	cur_level[cur_level_count] = node;
	cur_level_count++;

	while (cur_level_count > 0) {
		// printf("\n\n\nCUR_LEVEL_COUNT: %d\n", cur_level_count);
		// int bla;
		// printf("NODES: [");
		// for (bla = 0; bla < cur_level_count; bla++) {
		// 	printf("%d, ", weight(cur_level[bla]));
		// }
		// printf("]\n\n");

		if (cur_level_count < omp_get_max_threads() * 3) { //run sequential
			// printf("\nSTART OF SEQUENTIAL\n");
			node_t *cur_node;
			group_t *group;
			dataidxs_t *left; 
			dataidxs_t *right;
			node_t *temp_node;
			for (i = 0; i < cur_level_count; i++) {
				cur_node = cur_level[i];
				// printf("%d, ", weight(cur_node));
				group = cur_node->group; 
				left = group->left_idxs; 
				right = group->right_idxs;
				// printf("	NODE_TOTAL_N_ENTRIES: %d\n", left->n_entries + right->n_entries);
				if (left->n_entries == 0 || right->n_entries == 0) {
					if (left->n_entries == 0) {
						temp_node = create_leaf(train_set, right, cur_node->depth, n_features);
					}
					else {
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
			// printf("\nEND OF SEQUENTIAL\n");

		}
		else { //run in parallel
			// printf("\n\n\nSTART OF PARALLEL~~~~\n");
			quickSort(cur_level, 0, cur_level_count - 1);

			memset(cur_weights, 0, omp_get_max_threads() * sizeof(int));
			memset(cur_counts, 0, omp_get_max_threads() * sizeof(int));

			for (i = cur_level_count - 1; i > -1; i--) {
				min_weight = INT_MAX;
				for (j = 0; j < omp_get_max_threads(); j++) {
					if (cur_weights[j] < min_weight) {
						min_weight = cur_weights[j];
						min_thread_idx = j;
					}
				}

				cur_idxs[min_thread_idx][cur_counts[min_thread_idx]] = i;
				cur_counts[min_thread_idx]++;
				if (cur_counts[min_thread_idx] >= 1000) {
					fprintf(stderr, "ERROR in split(): did not allocate enough\n"); 
					exit(1);
				}
				cur_weights[min_thread_idx] += weight(cur_level[i]);
			}

			#pragma omp parallel
			{

				int t_count;
				int tid = omp_get_thread_num();
				// printf("HEREEEEEEEEEE TID = %d\n", tid);

				node_t *cur_node;
				group_t *group;
				dataidxs_t *left; 
				dataidxs_t *right;
				node_t *temp_node;
				for (t_count = 0; t_count < cur_counts[tid]; t_count++) {
					// printf("HEREEEEEEEEEE %d\n", t_count);

					cur_node = cur_level[cur_idxs[tid][t_count]];
					// printf("%d, ", weight(cur_node));
					group = cur_node->group; 
					left = group->left_idxs; 
					right = group->right_idxs;
					// printf("	NODE_TOTAL_N_ENTRIES: %d\n", left->n_entries + right->n_entries);
					if (left->n_entries == 0 || right->n_entries == 0) {
						if (left->n_entries == 0) {
							temp_node = create_leaf(train_set, right, cur_node->depth, n_features);
						}
						else {
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
						#pragma omp critical (next)
						{
							next_level[next_level_count] = cur_node->left;
							next_level_count++;
							if (next_level_count >= max_capacity) printf("\n\nERROR IN SPLIT: MAX CAPACITY REACHED\n\n");
						}
					}
					if (right->n_entries <= min_size) {
						cur_node->right = create_leaf(train_set, right, cur_node->depth + 1, n_features);
					}
					else {
						cur_node->right = get_split(train_set, right, n_features, cur_node->depth + 1);
						#pragma omp critical (next)
						{
							next_level[next_level_count] = cur_node->right;
							next_level_count++;
							if (next_level_count >= max_capacity) printf("\n\nERROR IN SPLIT: MAX CAPACITY REACHED\n\n");
						}
					}
				}
			}
			// printf("\nEND OF PARALLEL~~~~\n");
		}

		// printf("NEXT_LEVEL_COUNT: %d\n", next_level_count);
		// printf("BEFORE SORT: [");
		// for (i = 0; i < next_level_count; i++) {
		// 	printf("%d, ", next_level[i]->group->left_idxs->n_entries + next_level[i]->group->right_idxs->n_entries);
		// }
		// printf("]\n");


		// printf("AFTER SORT: [");
		// for (i = 0; i < next_level_count; i++) {
		// 	printf("%d, ", next_level[i]->group->left_idxs->n_entries + next_level[i]->group->right_idxs->n_entries);
		// }
		// printf("]\n\n\n");

		temp = cur_level;
		cur_level_count = next_level_count;
		cur_level = next_level;
		next_level_count = 0;
		next_level = temp;
	}

	free(cur_level);
	free(next_level);
	for (i = 0; i < omp_get_max_threads(); i++) {
		free(cur_idxs[i]);
	}
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

	for (i = 0; i < n_trees; i++) {
		prediction = predict(tree_list[i], test_set, row);
		if (prediction == 0) predict_0++; 
		else if (prediction == 1) predict_1++; 
	}

	return (predict_0 > predict_1) ? 0 : 1; 
}


float random_forest(float **train_set, float **test_set, int train_len, int test_len,
				   int max_depth, int min_size, float percentage,
				   int n_trees, int n_features) {

	double startSingleTree; 
	double endSingleTree; 

	tree_t **tree_list = malloc(n_trees * sizeof(tree_t*)); 

	int tree_index;

	double startSeconds = currentSeconds();

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

    FILE* stream;
	switch (file_size) {
		case 0: // xsmall
    		stream = fopen("../data/clean_data.csv", "r");
    		break;
		case 1: // small
    		stream = fopen("../data/x5_clean_data.csv", "r");
    		break;
    	case 2: // medium
    		stream = fopen("../data/medium_clean_data.csv", "r");
    		break;
    	case 3: // large
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
    					30, 							// max depth
    					10,								// min size
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
