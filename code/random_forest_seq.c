/*  Sequential Version of the Random Forest Algorithm */ 

#include <stdio.h>
#include <malloc.h>
#include <stdbool.h>
#include <limits.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int NUM_ENTRIES_TOTAL = 569;
int NUM_TEST_ENTRIES = 85; // Approx 15%
int NUM_TRAIN_ENTRIES = 484; 
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


stack_t *create_stack() {
	stack_t *stack = malloc(sizeof(stack_t)); 
	stack->end = 0; 
	stack->capacity = STACK_CAPACITY; 
	stack->nodeList = malloc(sizeof(node_t*) * stack->capacity); 
	return stack; 
}

bool stack_isEmpty(stack_t *stack) {
	return stack->end == 0; 
}

void stack_push(stack_t *stack, node_t *node) {
	if (stack->end == stack->capacity) {
		printf("Failed to Push onto Stack \n"); 
		return; 
	}
	stack->nodeList[stack->end] = node; 
	stack->end = stack->end + 1; 
}

node_t *stack_pop(stack_t *stack) {
	if (stack->end == 0) return NULL; 
	stack->end = stack->end - 1; 
	return stack->nodeList[stack->end + 1]; 
}

queue_t *create_queue() {
	queue_t *queue = malloc(sizeof(queue_t)); 
	queue->start = 0; 
	queue->end = 0; 
	queue->capacity = QUEUE_CAPACITY; 
	queue->nodeList = malloc(sizeof(node_t*) * queue->capacity); 
	return queue; 
}

bool queue_isEmpty(queue_t *queue) {
	return (queue->start == queue->end) && !(((queue->end + 1) % queue->capacity) == queue->start); 
}

void queue_push(queue_t *queue, node_t *node) {
	if(((queue->end + 1) % queue->capacity) == queue->start) {
		printf("Failed to Push onto Queue \n"); 
		return; 
	}
	queue->nodeList[queue->end] = node; 
	queue->end = (queue->end + 1) % queue->capacity; 
}

node_t *queue_pop(queue_t *queue) {
	if (queue->end == queue->start) return NULL; 
	node_t *node = queue->nodeList[queue->start]; 
	queue->start = (queue->start + 1) % queue->capacity; 
	return node; 
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


// Tested 
/* Create a random subsample from the dataset with replacement */
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


// Tested 
float gini_index(float **train_set, group_t *group) {
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
			if((int)train_set[index][NUM_FEATURES] == 0) {
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
			if(train_set[index][NUM_FEATURES] == 0.0) {
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

// Tested 
group_t *test_split(int index, float value, float **train_set, dataidxs_t *dataset) {
	dataidxs_t *left = create_dataidxs(NUM_TRAIN_ENTRIES); 
	dataidxs_t *right = create_dataidxs(NUM_TRAIN_ENTRIES); 

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

	// printf("FeatureList: \n"); 
	
	for(i = 0; i < n_features; i++){
		featureList[i] = i;
	}
	
	// featureList[0] = 0; 
	// featureList[1] = 2; 
	// featureList[2] = 4; 
	// featureList[3] = 6;
	// featureList[4] = 8; 
	// featureList[5] = 10; 
	// featureList[6] = 12; 

	// featureList[0] = 2; 
	// featureList[1] = 3; 
	// featureList[2] = 1; 
	// featureList[3] = 0;




	// Selecting the best split with the lowest gini index 
	for (index = 0; index < n_features; index++) {
		for (indexD = 0; indexD < dataset->n_entries; indexD++) {
			int feature_index = featureList[index]; 
			int data_index = (dataset->data_idxs)[indexD]; 

			group_t *group = test_split(feature_index, train_set[data_index][feature_index], train_set, dataset); 
			gini = gini_index(train_set, group); 
			if (gini < best_score) {
				best_feature_index = feature_index; 
				best_feature_value = train_set[data_index][feature_index]; 
				best_score = gini; 
				best_group = group; 
			} 
		}
	}

	node_t *node = create_node(best_feature_index, best_feature_value, best_group, node_depth); 

	return node; 
}

node_t *create_leaf(float **train_set, dataidxs_t *dataset, int node_depth) {
	int yes_count = 0; 
	int no_count = 0; 
	int i; 
	for (i = 0; i < dataset->n_entries; i++) {
		int index = dataset->data_idxs[i]; 
		if(train_set[index][NUM_FEATURES] == 1.0){
			yes_count++; 
		}
		else if(train_set[index][NUM_FEATURES] == 0.0) {
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
	int i;
	node_t *cur_node;
	group_t *group;
	dataidxs_t *left; 
	dataidxs_t *right;
	int max_capacity = 10000;
	node_t **temp;
	node_t **cur_level = malloc(sizeof(node_t*) * max_capacity);
	node_t **next_level = malloc(sizeof(node_t*) * max_capacity);
	int cur_level_count = 0;
	int next_level_count = 0;

	cur_level[cur_level_count] = node;
	cur_level_count++;

	while (cur_level_count > 0) {
		for (i = 0; i < cur_level_count; i++) {
			cur_node = cur_level[i];
			group = cur_node->group; 
			left = group->left_idxs; 
			right = group->right_idxs;
			if (left->n_entries == 0 || right->n_entries == 0) {
				int result;
				if (left->n_entries == 0) {
					result = create_leaf(train_set, right, cur_node->depth)->result;
				}
				else if (right->n_entries == 0) {
					result = create_leaf(train_set, left, cur_node->depth)->result;
				}
				cur_node->leaf = true;
				cur_node->result = result;
				continue;
				// Free the created leaf
			}

			if (cur_node->depth >= max_depth - 1) {
				cur_node->left = create_leaf(train_set, left, cur_node->depth + 1);
				cur_node->right = create_leaf(train_set, right, cur_node->depth + 1);
				continue;
			}

			if (left->n_entries <= min_size) {
				cur_node->left = create_leaf(train_set, left, cur_node->depth + 1);
			}
			else {
				cur_node->left = get_split(train_set, left, n_features, cur_node->depth + 1);
				next_level[next_level_count] = cur_node->left;
				next_level_count++;
				if (next_level_count >= max_capacity) printf("\n\nERROR IN SPLIT: MAX CAPACITY REACHED\n\n");
			}
			if (right->n_entries <= min_size) {
				cur_node->right = create_leaf(train_set, right, cur_node->depth + 1);
			}
			else {
				cur_node->right = get_split(train_set, right, n_features, cur_node->depth + 1);
				next_level[next_level_count] = cur_node->right;
				next_level_count++;
				if (next_level_count >= max_capacity) printf("\n\nERROR IN SPLIT: MAX CAPACITY REACHED\n\n");
			}
		}

		temp = cur_level;
		cur_level_count = next_level_count;
		cur_level = next_level;
		next_level_count = 0;
		next_level = temp;
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
	for (i = 0; i < n_trees; i++) {
		prediction = predict(tree_list[i], test_set, row); 
		printf("Prediction: %d\n", prediction); 
		if (prediction == 0) predict_0++; 
		else if (prediction == 1) predict_1++; 
	}
	printf("\n"); 
	return (predict_0 > predict_1) ? 0 : 1; 
}


// Dataset should be int **
int* random_forest(float **train_set, float **test_set, int train_len, int test_len,
				   int max_depth, int min_size, float percentage,
				   int n_trees, int n_features) {

	tree_t **tree_list = malloc(n_trees * sizeof(tree_t*)); 

	int tree_index; 

	for (tree_index = 0; tree_index < n_trees; tree_index++) {
		dataidxs_t *sample = subsample(train_len, percentage); 
		node_t *root_node = build_tree(train_set, sample, max_depth, min_size, n_features); 
		tree_t *tree = malloc(sizeof(tree_t)); 
		tree->root_node = root_node; 
		tree_list[tree_index] = tree; 
	}


	int row; 
	int *predictions = malloc(test_len * sizeof(int)); 
	for (row = 0; row < test_len; row++) {
		predictions[row] = bagging_predict(tree_list, n_trees, test_set, row); 
	}
	return predictions; 
}


float* get_row(char* line, int num)
{
    const char* tok = strtok(line, ",");
    float *arr = malloc(sizeof(float) * 40);
    int i;
    for (i = 0; i < num; i++) {
    	// printf(tok);
    	// printf("\n");
    	arr[i] = atof(tok);
    	tok = strtok(NULL, ",");
    }
    // for (tok = strtok(line, ","); tok && *tok; tok = strtok(NULL, ",\n"))
    // {
    // 	num--;
    // 	arr[]
    //     if (!num) break;
    // }
    return arr;
}

int main()
{
    FILE* stream = fopen("../data/clean_data.csv", "r");
    float **data = malloc(sizeof(float *) * 10000);
    char line[4096];
    int count = 0;
    while (fgets(line, 4096, stream))
    {
        char* tmp = strdup(line);
       	int n = 31;
       	float *arr = get_row(tmp, n);
       	data[count] = arr;
       	count++;
        // NOTE strtok clobbers tmp
        free(tmp);
    }


    
}
