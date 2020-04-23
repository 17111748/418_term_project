/*  Sequential Version of the Random Forest Algorithm */ 

#include <stdio.h>
#include <malloc.h>
#include <stdbool.h>
#include <limits.h>
#include <stdlib.h>
#include <math.h>

int NUM_ENTRIES_TOTAL = 569;
int NUM_TEST_ENTRIES = 85; // Approx 15%
int NUM_TRAIN_ENTRIES = 484; 
int NUM_FEATURES = 30;

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


/* Create a random subsample from the dataset with replacement */
dataidxs_t* subsample(int n_entries, float percentage) {
	int i;
	int n_sample = (float)(n_entries * percentage) * 10 / 10; 
	dataidxs_t *sample = create_dataidxs(n_sample);
	for (i = 0; i < n_sample; i++) {
		int index = rand() % n_entries; 
		sample->data_idxs[i] = index; 
	}
	return sample; 
}

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
	size = (float)(right_idxs->n_entries); 
	if (size != 0) {
		score = 0.0; 
		p0 = 0.0; 
		p1 = 0.0; 
		for (k = 0; k < right_idxs->n_entries; k++) {
			int index = right_idxs->data_idxs[k]; 
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

group_t *test_split(int index, float value, float **train_set) {
	dataidxs_t *left = create_dataidxs(NUM_TRAIN_ENTRIES); 
	dataidxs_t *right = create_dataidxs(NUM_TRAIN_ENTRIES); 
	int left_count = 0;
	int right_count = 0; 

	int row; 
	
	for (row = 0; row < NUM_TRAIN_ENTRIES; row++) {
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

	// Selecting the best split with the lowest gini index 
	for (index = 0; index < n_features; index++) {
		for (indexD = 0; indexD < dataset->n_entries; indexD++) {
			int feature_index = featureList[index]; 
			int data_index = (dataset->data_idxs)[indexD]; 

			group_t *group = test_split(feature_index, train_set[data_index][feature_index], train_set); 
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
	int no_count = 1; 
	int i; 
	for (i = 0; i < dataset->n_entries; i++) {
		int index = dataset->data_idxs[i]; 
		if(train_set[index][NUM_FEATURES-1] == 1.0){
			yes_count++; 
		}
		else if(train_set[index][NUM_FEATURES-1] == 0.0) {
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
	group_t *group = node->group; 
	dataidxs_t *left = group->left_idxs; 
	dataidxs_t *right = group->right_idxs;

	stack_t *stack = malloc(sizeof(stack_t)); 

	node_t *root = node; 

	// root->leaf means root != NULL 
	while (!root->leaf) {
		int depth = root->depth; 
		if (left->n_entries == 0) {
			root->left = create_leaf(train_set, left, depth + 1); 
			root->right = create_leaf(train_set, left, depth + 1); 
		}
		else if (right->n_entries == 0) {
			root->left = create_leaf(train_set, right, depth + 1); 
			root->right = create_leaf(train_set, right, depth + 1); 
		}
		else if (depth >= max_depth) {
			root->left = create_leaf(train_set, left, depth + 1); 
			root->right = create_leaf(train_set, right, depth + 1); 
		}
		else {
			if (left->n_entries > min_size) {
				root->left = get_split(train_set, left, n_features, depth + 1);  // Fix what the get_split takes in
			}
			else {
				root->left = create_leaf(train_set, left, depth + 1); 
			}

			if (right->n_entries > min_size) {
				root->right = get_split(train_set, right, n_features, depth + 1); 
				stack_push(stack, root->right); 
			}

		}

		// Update the value of the root 
		if (!root->left->leaf){
			root = root->left; 
		}
		else {
			root = stack_pop(stack); 
		}
		left = root->group->left_idxs; 
		right = root->group->right_idxs; 
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


// Dataset should be int **
int* random_forest(float **train_set, float **test_set, int train_len, int test_len,
				   int max_depth, int min_size, int percentage,
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


int main() {
	return 0; 
}





// public static Tree builtBSTFromSortedArray(int[] inputArray){

//     Stack toBeDone=new Stack("sub trees to be created under these nodes");

//     //initialize start and end 
//     int start=0;
//     int end=inputArray.length-1;

//     //keep memoy of the position (in the array) of the previously created node
//     int previous_end=end;
//     int previous_start=start;

//     //Create the result tree 
//     Node root=new Node(inputArray[(start+end)/2]);
//     Tree result=new Tree(root);
//     while(root!=null){
//         System.out.println("Current root="+root.data);

//         //calculate last middle (last node position using the last start and last end)
//         int last_mid=(previous_start+previous_end)/2;

//         //*********** add left node to the previously created node ***********
//         //calculate new start and new end positions
//         //end is the previous index position minus 1
//         end=last_mid-1; 
//         //start will not change for left nodes generation
//         start=previous_start; 
//         //check if the index exists in the array and add the left node
//         if (end>=start){
//             root.left=new Node(inputArray[((start+end)/2)]);
//             System.out.println("\tCurrent root.left="+root.left.data);
//         }
//         else
//             root.left=null;
//         //save previous_end value (to be used in right node creation)
//         int previous_end_bck=previous_end;
//         //update previous end
//         previous_end=end;

//         //*********** add right node to the previously created node ***********
//         //get the initial value (inside the current iteration) of previous end
//         end=previous_end_bck;
//         //start is the previous index position plus one
//         start=last_mid+1;
//         //check if the index exists in the array and add the right node
//         if (start<=end){
//             root.right=new Node(inputArray[((start+end)/2)]);
//             System.out.println("\tCurrent root.right="+root.right.data);
//             //save the created node and its index position (start & end) in the array to toBeDone stack
//             toBeDone.push(root.right);
//             toBeDone.push(new Node(start));
//             toBeDone.push(new Node(end));   
//         }

//         //*********** update the value of root ***********
//         if (root.left!=null){
//             root=root.left; 
//         }
//         else{
//             if (toBeDone.top!=null) previous_end=toBeDone.pop().data;
//             if (toBeDone.top!=null) previous_start=toBeDone.pop().data;
//             root=toBeDone.pop();    
//         }
//     }
//     return result;  
// }
