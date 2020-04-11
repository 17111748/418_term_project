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
	node_t *rootNode; 
} tree_t

typedef struct {
	int feature; 
	node_t *left; 
	node_t *right; 
	int depth;
} node_t

typedef struct {
	float **left; 
	float **right; 
	int left_len;
	int right_len; 
} group_t

typedef struct {
	float **data;
	int num_rows; 
} dataset_t


node_t* create_node(int data, int level) {
	node_t *node = malloc(sizeof(node_t)); 
	node->left = NULL; 
	node->right = NULL; 
	node->data = data; 
	node->level = level; 
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


node_t build_tree(int **sample, int max_depth, int min_size, int n_features) {
	node_t *rootNode; 





	return rootNode; 
}



// Dataset should be int **
int* random_forest(int **train, int **test, int train_len, int test_len, int max_depth, int min_size, 
					int sample_size, int n_trees, int n_features) {

	tree_t *treeList = malloc(n_trees * sizeof(tree_t)); 

	int tree_index; 

	for (tree_index = 0; tree_index < n_trees; tree_index++) {
		int **sample = subsample(train, train_len, sample_size); 
		node_t *rootNode = build_tree(sample, max_depth, min_size, n_features); 
		tree_t *tree = malloc(sizeof(tree_t)); 
		tree->rootNode = rootNode; 
		treeList[tree_index] = tree; 
	}

	// Assume predictions are integers 
	int row; 
	int *predictions = malloc(test_len * sizeof(int)); 
	for(row = 0; row < test_len; row++) {
		predictions[row] = bagging_predict(treeList, test, row); 
	}
	return predictions; 
}




// TODO: Check Dataset type 
data_t* subsample(data_t *dataset, int dataset_len, float ratio) {
	int n_sample = round((float)dataset_len * ratio); 
	int i; 
	for (i = 0; i < n_sample; i++) {
		int index = rand() % dataset_len; 
		sample[i] = dataset[index]; 
	}
	return sample;  
}

int predict(tree_t *treeList, int treeList_len, int *test, int test_len, int row) {
	if()
}

// TODO: Find maximum occurence 
int bagging_predict(tree_t *treeList, int treeList_len, int *test, int test_len, int row) {
	int i; 
	int *predictions[treeList_len]; 
	for (i = 0; i < treeList_len; i++) {
		predictions[i] = predict(treeList, treeList_len, test, test_len, row); 
	}

	int prediction; 
	// TODO: Find the maximum occurance 

	return prediction; 
}


