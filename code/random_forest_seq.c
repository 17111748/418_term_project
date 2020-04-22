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
	float feature_value;
	int feature;
	int depth;
	int result;  
} node_t;

typedef struct {
	dataidxs_t *left_idxs; 
	dataidxs_t *right_idxs;
} group_t;

typedef struct {
	int *data_idxs;
	int n_entries;
} dataidxs_t;


node_t* create_node(int feature, float feature_value, int depth) {
	node_t *node = malloc(sizeof(node_t)); 
	node->left = NULL; 
	node->right = NULL; 
	node->feature = feature; 
	node->depth = depth;
	node->feature_value = feature_value; 
	node->result = -1; 
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

group_t *test_split(int index, float value, float **train_set) {
	dataidxs_t *left = create_dataidxs(NUM_TRAIN_ENTRIES); 
	dataidxs_t *right = create_dataidxs(NUM_TRAIN_ENTRIES); 
	int left_count = 0;
	int right_count = 0; 
	
	for (int row = 0; row < NUM_TRAIN_ENTRIES; row++) {
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

node_t *get_split(float **train_set, int n_features, int node_depth) {
	int best_feature_index = -1; 
	float best_feature_value = -1; 
	float best_score = (float)INT_MAX; 
	group_t *best_group = NULL; 

	int index; 
	int count; 

	// Randomly Select N features from featureList 
	int featureList[n_features]; 
	featureList[0] = rand() % n_features; 
	for (int i = 1; i < n_features; i++) {
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

	// Selecting the best split with the lowest gini index 
	for (int feature_index = 0; feature_index < n_features; feature_index++) {
		for (int data = 0; data < NUM_TRAIN_ENTRIES; data++) {
			group_t *group = test_split(feature_index, train_set[data][feature_index], train_set); 
			float gini = gini_index(train_set, group); 
			if (gini < best_score) {
				best_feature_index = feature_index; 
				best_feature_value = train_set[data][feature_index]; 
				best_score = gini; 
				best_group = group; 
			} 
		}
	}

	node_t *node = create_node(best_feature_index, best_feature_value, node_depth); 

	return node; 
}

void *split(node_t *node, int max_depth, int min_size, int n_features, float **train_set) {

}


node_t *build_tree(float **train_set, dataidxs_t *sample, int max_depth, int min_size, 
					int n_features) {

	node_t *root_node = get_split(train_set, n_features, 0);
	split(root_node, max_depth, min_size, n_features, train_set); 

	return root_node; 
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
		predictions[row] = bagging_predict(tree_list, n_trees, test_set, row); 
	}
	return predictions; 
}



// Pass in n_trees
int predict(tree_t tree, float **test_set, int row) {
	node_t *cur_node = tree->root_node; 
	float *test_row = test_set[row]; 
	int feature; 
	float feature_value; 
	if (cur_node == NULL) return -1; 
	while ((cur_node->left == NULL) && (cur_node->right == NULL)) {
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
int bagging_predict(tree_t *tree_list, int n_trees, float **test_set, int row) {
	int i; 
	int prediction; 
	int predict_0 = 0;  
	int predict_1 = 0;  
	for (i = 0; i < n_trees; i++) {
		prediction = predict(tree_list[i], test, row); 
		if (prediction == 0) predict_0++; 
		else if (prediction == 1) predict_1++; 
	}

	return (predict_0 > predict_1) ? 0 : 1; 
}



float gini_index(float **train_set, group_t *group) {
	dataidxs_t *left_idxs = group->left_idxs; 
	dataidxs_t *right_idxs = group->right_idxs; 

	int n_instances = left_idxs->n_entries + right_idxs->n_entries; 
	float gini = 0.0; 
	int i, j, k; 
	float size, score, p0, p1; 

	// Left Side 
	size = float(left_idxs->n_entries); 
	if (size != 0) {
		score = 0.0; 
		p0 = 0.0; 
		p1 = 0.0; 
		for (k = 0; k < left_idxs->n_entries; k++) {
			int index = left_idxs[k]; 
			// get malign or not count 
			if((int)train_set[index][NUM_FEATURES - 1] == 0) {
				p0 += 1; 
			}
			else {
				p1 += 1; 
			}
		}
		score += p0/size * p0/size;
		gini += (1.0 - score) * (size / n_instances); 
	}

	// Right Side 
	size = float(right_idxs->n_entries); 
	if (size != 0) {
		score = 0.0; 
		p0 = 0.0; 
		p1 = 0.0; 
		for (k = 0; k < right_idxs->n_entries; k++) {
			int index = right_idxs[k]; 
			if((int)train_set[index][NUM_FEATURES - 1] == 0) {
				p0 += 1; 
			}
			else {
				p1 += 1; 
			}
		}
		score += p0/size * p0/size;
		gini += (1.0 - score) * (size / n_instances); 
	}
	
	return gini; 
}

