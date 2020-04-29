'''
1. Across the forest
    - use large number for n_trees
    - split the work done for each tree
    - e.g. each thread builds x number of trees


2. Across a single tree
    - probably need to generate a more messy and larger dataset to test
    - split the work done for each node by each level
    - top few levels (aka not that many nodes) --> seq
    - e.g. each thread builds x number of nodes in the tree


3. Across a single node
    - split the work done within a node
    - e.g. parallelizing the three for loops (2 in get_split, 1 in test_split)
    Loops:
        a. n_features (constant)
        b. dataset->n_entries (constant for a single node->left/right)
        c. dataset->n_entries (constant for a single node->left/right)
    ideally, a * b * b / t no. iterations each
    try: pragma for static on each of these loops

4. Classifying the dataset
    - split the work done for each entry
    - e.g. each thread classifies x number of entries


left->n_entries + right->n_entries

[10000] 
[9000] [1000] (2 threads)
[8000][1000][500][500] <--- 4 nodes

[500][1000][8000][500]

0. -
1. [500]
2. [1000][500]
3. [8000][1000][500]
4. [8000][1000][500][500]

partition --> snake

0       1
8000    1000
500     500


500 nodes
1. sort
2. 

'''


nfeat = 20
nentries = 10

total = nfeat*nentries

nthreads = 8

print("Total:", total)

# curthread = 0
curthread = 7
n_per_thread = total//nthreads
extra_threads = total%nthreads

print("N per thread:", n_per_thread)
print("Extra_threads:", extra_threads)

for curthread in range(nthreads):
    for i in range(nfeat):
        for j in range(nentries):
            cur_idx = i*nentries+j
            min_idx = curthread*n_per_thread
            max_idx = (1+curthread)*n_per_thread
            if (min_idx <= cur_idx and cur_idx < max_idx):
                print(cur_idx)
            elif (cur_idx >= n_per_thread * nthreads and
                curthread == total - cur_idx - 1):
                print(cur_idx)
    print()