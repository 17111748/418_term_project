#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void delay(int number_of_seconds) 
{ 
    // Converting time into milli_seconds 
    int milli_seconds = 1000 * number_of_seconds; 
  
    // Storing start time 
    clock_t start_time = clock(); 
  
    // looping till required time is not achieved 
    while (clock() < start_time + milli_seconds) 
        continue; 
}

int main(int argc, char *argv[]) {
    clock_t start = clock();
    clock_t diff;
    int i;
    int g = 0;
    for (i = 0; i < 100000; i++) {
        g += 1;
    }
    diff = clock() - start;

    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);
    
    return 0;
}

// int allocs_in_use;
// void my_free(void * p) {
//     allocs_in_use--;
//     free(p);
// }
// void * my_alloc(int size) {
//     void * p = malloc(size);
//     allocs_in_use++;
//     if (p == NULL) {
//         fprintf(stderr, "Ran out of space.  Requested size=%d.\n", size);
//         exit(1);
//     }
//     return p;
// }

// /* used internally by the LP solver  */
// int m, n;         /* number of constraints and variables */
// int* non_basic;   /* indices of non-basic variables.  Length=n */
// int* basic;       /* indices of basic variables.  Length=m */
// double** a;       /* (m+1) x (n+1) tableau for simplex */

// #define INF 1e100
// #define EPS 1e-9

// void pivot(int r, int c) {
//     int i,j,t;
//     t = non_basic[c];
//     non_basic[c] = basic[r];
//     basic[r] = t;

//     a[r][c]=1/a[r][c];
//     for(j=0;j<=n;j++) if(j!=c) a[r][j]*=a[r][c];
//     for(i=0;i<=m;i++) if(i!=r) {
// 	    for(j=0;j<=n;j++) if(j!=c) a[i][j]-=a[i][c]*a[r][j];
// 	    a[i][c] = -a[i][c]*a[r][c];
// 	}
// }

// int feasible() {
//     int r,c,i; double p,v;
//     while(1) {
// 	for(p=INF,i=0; i<m; i++) if(a[i][n]<p) p=a[r=i][n];
// 	if(p>-EPS) return 1;
// 	for(p=0,i=0; i<n; i++) if(a[r][i]<p) p=a[r][c=i];
// 	if(p>-EPS) return 0;
// 	p = a[r][n]/a[r][c];
// 	for(i=r+1; i<m; i++) if(a[i][c]>EPS) {
// 		v = a[i][n]/a[i][c];
// 		if(v<p) r=i,p=v;
// 	    }
// 	pivot(r,c);
//     }
// }

// int simplex(int m0, int n0, double* A[], double B[], double C[], double* z, double soln[]) {
//     /*
//     input:
//       m = #constraints, n =#variables
//       max C dot x s.t. A x <= B
//       where A = mxn, B = m vector, C = n vector
//     output:
//       returns 1 (feasible), 0 (infeasible), or -1 (unbounded)
//       If feasible, then stores objective value in z, and the solution in soln,
//       an array of length n supplied for the variables.
//     caveats:
//       Cycling is possible.  Nothing is done to mitigate loss of
//       precision when the number of iterations is large.
//     */
//     int r,c,i,j;
//     double p, v;

//     m = m0;
//     n = n0;

//     non_basic = (int *) my_alloc(n * sizeof(int));
//     basic = (int *) my_alloc(m * sizeof(int));
//     for(i=0; i<n; i++) non_basic[i]=i;
//     for(i=0; i<m; i++) basic[i]=n+i;

//     a = (double **) my_alloc((m+1) * sizeof (double *));
    
//     for (i=0; i<=m; i++) {
// 	a[i] = (double *) my_alloc((n+1) * sizeof (double));
//     }
    
//     for(i=0; i<m; i++) {
// 	for(j=0; j<n; j++) {
// 	    a[i][j] = A[i][j];
// 	};
// 	a[i][n] = B[i];
//     }
//     for(j=0; j<n; j++) {
// 	a[m][j] = C[j];
//     }

//     a[m][n] = 0.0;
    
//     if(!feasible()) return 0;

//     while(1) {
// 	for(p=0,i=0; i<n; i++) if(a[m][i]>p) p=a[m][c=i];
// 	if(p<EPS) {
// 	    for(i=0; i<n; i++) if(non_basic[i]<n) soln[non_basic[i]]=0.0;
// 	    for(i=0; i<m; i++) if(basic[i]<n) soln[basic[i]]=a[i][n];
// 	    *z = -a[m][n];
// 	    return 1;
// 	}
// 	for(p=INF,i=0; i<m; i++) if(a[i][c]>EPS) {
// 		v = a[i][n]/a[i][c];
// 		if(v<p) p=v,r=i;
// 	    }
// 	if(p==INF) return -1;
// 	pivot(r,c);
//     }
// }


// typedef struct node {
//     int a;
//     int b;
// } node_t; 


// void print_node_list(node_t ** node_list, int len) {
//     printf("\n\n~~~~~ START OF PRINTING ~~~~~~~\n");
//     int i;
//     for (i = 0; i < len; i++) {
//         printf("a: %d\n", node_list[i]->a);
//         printf("b: %d\n\n", node_list[i]->b);
//     }

//     printf("~~~~~ END OF PRINTING ~~~~~~~\n\n");

// }

// int main(int argc, char *argv[]) {
//     int max_capacity = 100;

//     node_t **cur_level = malloc(sizeof(node_t*) * max_capacity);
//     node_t **next_level = malloc(sizeof(node_t*) * max_capacity);
//     int cur_level_len = 6;
//     int next_level_len = 3;
//     int i;
//     for (i = 0; i < cur_level_len; i++) {
//         node_t *n = malloc(sizeof(node_t));
//         n->a = i;
//         n->b = i*i;
//         cur_level[i] = n;
//     }

//     for (i = 0; i < next_level_len; i++) {
//         node_t *n = malloc(sizeof(node_t));
//         n->a = i+1000;
//         n->b = i+1001;
//         next_level[i] = n;
//     }

//     print_node_list(cur_level, cur_level_len);
//     print_node_list(next_level, next_level_len);

//     node_t **temp = cur_level;
//     cur_level = next_level;
//     cur_level_len = next_level_len;

//     next_level = temp;
//     next_level_len = 4;
//     for (i = 0; i < next_level_len; i++) {
//         node_t *n = malloc(sizeof(node_t));
//         n->a = i+1234;
//         n->b = i+1+1234;
//         next_level[i] = n;
//     }

//     print_node_list(cur_level, cur_level_len);
//     print_node_list(next_level, 6);

//     // tree_t **tree_list = (tree_t **)malloc(n_trees * sizeof(tree_t *)); 

//     // for (int i = 0; i < n_trees; i++) {
//     //     tree_t *tree = (tree_t *)malloc(sizeof(tree_t));
//     //     tree->root_node = get_root_node(...); 
//     //     tree_list[i] = tree;
//     // }

//  //    double A0[] = {1.0, 3.0, 1.0};
//  //    double A1[] = {-1.0, 0.0, 3.0};
//  //    double A2[] = {2.0, -1.0, 2.0};
//  //    double A3[] = {2.0, 3.0, -1.0};
//  //    double*A[] = {A0, A1, A2, A3};

//  //    double B[] = {3.0, 2.0, 4.0, 2.0};

//  //    double C[] = {5.0, 5.0, 3.0};

//  //    int m = 4;
//  //    int n = 3;
//  //    int i;
//  //    double z;
//  //    double* soln = (double *) my_alloc(n * sizeof (double));

//  //    int ret = simplex(m, n, A, B, C, &z, soln);

//  //    if (ret == -1) {
//  //        printf("unbounded\n");
//  //    } else if (ret == 0) {
//  //        printf ("infeasible\n");	
//  //    } else if (ret == 1) {
// 	// printf("The optimum is: %f\n", z);
// 	// for (i = 0; i<n; i++) {
// 	//     printf("x%d = %f\n", i, soln[i]);
// 	// }
//  //    } else {
// 	// printf("Should not have happened\n");
//  //    }

//  //    my_free(basic);
//  //    my_free(non_basic);
//  //    my_free(soln);
//  //    for (i=0; i<=m; i++) my_free(a[i]);
//  //    my_free(a);
//  //    printf("allocs_in_use = %d\n", allocs_in_use);

//     int data[100][15];
//     int i,j;
//     for (i = 0; i<100; i++) {
//         for (j = 0; j < 15; j++) {
//             data[i][j] = (((i*13)+(j*829))%97)/100
//         }
//     }
    
//     return 0;
// }
