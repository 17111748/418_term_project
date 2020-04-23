#include <stdio.h>
#include <stdlib.h>
#include "random_forest_seq.c"


void print_dataidxs_t(dataidxs_t *d) {
	printf("\n~~~~~~~~ Print DataIdxs_t ~~~~~~~~~~\n"); 
	printf("n_entries: %d \n", d->n_entries); 
	int i; 
	printf("indexs: "); 
	for (i = 0; i < d->n_entries; i++) {
		printf("%d, ", d->data_idxs[i]); 
	}
	printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"); 
}

void print_group_t(group_t *g) {
	printf("\n~~~~~~~~ Print Group_t ~~~~~~~~~~~~~\n"); 
	dataidxs_t *left = g->left_idxs; 
	dataidxs_t *right = g->right_idxs;
	
	printf("Left Data Index: \n"); 
	print_dataidxs_t(left); 
	printf("Right Data Index: \n"); 
	print_dataidxs_t(right); 
	
	printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"); 
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

void printPost(node_t *n){
	if(n == NULL) return; 
	print_node_t(n); 
	printPost(n->left); 
	printPost(n->right); 
}

int main(int argc, char *argv[]) {
	int n_features = 14; 
	int n_rows = 100; 
	float **train_set = malloc(sizeof(float *) * n_rows); 

	int i; 
	for (i = 0; i < n_rows; i++) {
		train_set[i] = malloc(sizeof(float) * (n_features + 1)); 
		// for (j = 0; j < n_features + 1; j++) {
	}

	
	int j;
	for (i = 0; i<100; i++) {
	    for (j = 0; j < 15; j++) {
	        train_set[i][j] = (float)(((i*13)+(j*829))%97)/ (float)100; 
	        if (j == 14) {
	        	train_set[i][j] = i % 2; 
	        }
	        // printf("i: %d, j: %d, val: %f\n", i, j, train_set[i][j]); 
	    }
	}

	// train_set[0][0] = 0.1;
	// train_set[1][0] = 0.15;
	// train_set[2][0] = 0.3;
	// train_set[3][0] = 1.2;
	// train_set[4][0] = 7; 

	// train_set[0][1] = 0.3;
	// train_set[1][1] = 7.4;
	// train_set[2][1] = 1.3;
	// train_set[3][1] = 5.2;
	// train_set[4][1] = 2.5;

	// train_set[0][2] = 7;
	// train_set[1][2] = 11;
	// train_set[2][2] = 8;
	// train_set[3][2] = 9; 
	// train_set[4][2] = 10;

	// train_set[0][3] = 9.1;
	// train_set[1][3] = 11;
	// train_set[2][3] = 8.4;
	// train_set[3][3] = 0.1; 
	// train_set[4][3] = -0.2;

	// train_set[0][4] = 1.0;
	// train_set[1][4] = 0.0;
	// train_set[2][4] = 1.0;
	// train_set[3][4] = 0.0; 
	// train_set[4][4] = 1.0;

	dataidxs_t *test = create_dataidxs(n_rows); 
	
	for (i= 0; i < n_rows; i++) {
		test->data_idxs[i] = i; 
	}

	// tree_t **tree_list = malloc(1 * sizeof(tree_t*)); 
	// node_t *root_node = build_tree(train_set, test, 10, 1, n_features); 
	// tree_t *tree = malloc(sizeof(tree_t)); 
	// tree->root_node = root_node; 		
	// tree_list[0] = tree;
	int test_count = 4; 

	float **test_set = malloc(sizeof(float *) * n_rows); 


	for (i = 0; i < test_count; i++) {
		test_set[i] = malloc(sizeof(float) * (n_features + 1)); 
		for (j = 0; j < n_features; j++) {
			test_set[i][j] = (3.7* i + 1.4*j) / 1.3; 
			printf("i: %d, j: %d, val: %f\n", i, j, test_set[i][j]); 
		}
	}


	// printf("predict: %d\n", predict(tree_list[0], test_set, 0));

	int *ans = random_forest(train_set, test_set, n_rows, test_count, 10, 1, 1.0, 3, n_features);  

	for (i = 0; i < test_count; i++) {
		printf("%dth Answer: %d\n", i, ans[i]); 
	}
	
	// print_dataidxs_t(subsample(21, 0.5)); 

	// float gini = gini_index(train_set, test_split(1, 0.4, train_set, test)); 
	// printf("gini: %f\n", gini); 



	// print_node_t(get_split(train_set, test, n_features, 0)); 


	// node_t *node = get_split(train_set, test, n_features, 0); 
	// split(node, 10, 1, n_features, train_set); 

	// printPost(node); 
	


    return 0;

}