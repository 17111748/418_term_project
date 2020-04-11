/*  Sequential Version of the Random Forest Algorithm */ 

#include <stdio.h>
#include <malloc.h>
#include <conio.h>
#include <stdlib.h>
#include <math.h>

int NUM_ENTRIES_TOTAL = 569;
int NUM_TEST_ENTRIES = 85; // Approx 15%
int NUM_TRAIN_ENTRIES = 484; 
int NUM_FEATURES = 30;

typedef struct {
	node_t *root_node; 
} tree_t;

typedef struct {
	node_t *left; 
	node_t *right; 
	int feature;
	int depth;
} node_t;

typedef struct {
	dataidxs_t *left_idxs; 
	dataidxs_t *right_idxs;
} group_t;

typedef struct {
	int *data_idxs;
	int n_entries;
} dataidxs_t;


node_t* create_node(int feature, int depth) {
	node_t *node = malloc(sizeof(node_t)); 
	node->left = NULL; 
	node->right = NULL; 
	node->feature = feature; 
	node->depth = depth;
	return node;
}

dataidxs_t* create_dataidxs(int n_entries) {
	dataidxs_t *dataidxs = malloc(sizeof(dataidxs_t)); 
	int *idxs = malloc(sizeof(int) * n_entries); 
	dataidxs->data_idxs = idxs;
	dataidxs->n_entries = n_entries; 
	return dataidxs;
}


/* Create a random subsample from the dataset with replacement */
dataidxs_t* subsample(int n_entries, float percentage) {
	int i;
	int n_sample = round((float)n_entries * percentage);
	dataidxs_t *sample = create_dataidxs(n_sample);
	for (i = 0; i < n_sample; i++) {
		int index = rand() % n_entries; 
		sample->data_idxs[i] = index; 
	}
	return sample; 
}


float gini_index(group_t group, int *labelList, int labelList_len) {
	int n_instances = group->left_len + group->right_len; 
	float gini = 0.0; 
	int i, j, k; 
	float size, score, p; 
	for (i = 0; i < 2; i++) {
		size = 2.0; 
		score = 0.0; 
		for (j = 0; j < labelList_len; j++) {
			p = 0.0; 
			if(i == 0) {
				for (k = 0; k < left_len; k++) {

				}
			}
			else {
				for (k = 0; k < right_len; k++) {

				}
			}
			score += p; 
		}
		gini += (1.0 - score) * (size / n_instances); 
	}
	return gini; 
}




node_t get_split(int **dataset, int n_features) {

}


node_t build_tree(float **train_set, dataidxs_t *sample, int max_depth, int min_size, int n_features) {
	node_t *rootNode; 



	return rootNode; 
}



// Dataset should be int **
int* random_forest(float **train_set, float **test_set, int train_len, int test_len,
				   int max_depth, int min_size, int percentage,
				   int n_trees, int n_features) {

	tree_t *tree_list = malloc(n_trees * sizeof(tree_t)); 

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
		predictions[row] = bagging_predict(tree_list, test_set, row); 
	}
	return predictions; 
}






int predict(tree_t *tree_list, int tree_list_len, int *test, int test_len, int row) {
	if()
}

// TODO: Find maximum occurence 
int bagging_predict(tree_t *tree_list, int tree_list_len, int *test, int test_len, int row) {
	int i; 
	int *predictions[tree_list_len]; 
	for (i = 0; i < tree_list_len; i++) {
		predictions[i] = predict(tree_list, tree_list_len, test, test_len, row); 
	}

	int prediction; 
	// TODO: Find the maximum occurance 

	return prediction; 
}


