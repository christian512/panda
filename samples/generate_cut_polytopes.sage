# Run this file by executing "sage generate_cut_polytopes.sage 1 4 4"

import sys

# Parse partition sizes from command line arguments
if len(sys.argv) < 2:
    print("Usage: sage generate_cut_polytopes.sage <partition_size_1> <partition_size_2> ...")
    print("Example: sage generate_cut_polytopes.sage 1 4 4")
    sys.exit(1)

import os

partition_sizes = [int(x) for x in sys.argv[1:]]
outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'panda_format', 'cut')
os.makedirs(outdir, exist_ok=True)
output_filename = os.path.join(outdir, 'CUT_' + '_'.join(str(x) for x in partition_sizes) + '.ext')

graph = graphs.CompleteMultipartiteGraph(partition_sizes)

vertices = graph.vertices()
gens = graph.automorphism_group().gens()
print("Partition sizes: ", partition_sizes)
print("Number of Graph Vertices: ", len(vertices))

# Define subgroups
vertices_set = sage.sets.set.Set_object(vertices)
subsets_list = []
for subset in vertices_set.subsets():
    diff = sorted(list(vertices_set.difference(subset)))
    if diff in subsets_list:
        continue
    subsets_list.append(sorted(list(subset)))

# Check that the number of subgroups is correct
print("Number of Polytope Vertices: ", len(subsets_list))
assert len(subsets_list) == 2**(len(vertices)-1)


# Open a file to write the output to
f = open(output_filename, 'w+')


# Generate polytope vertices
f.write('Vertices:\n')
def metric(x,y, subset):
    """ Metric to define the cut polytope """
    if not (x,y,None) in graph.edges():
        return 0
    intersection = list(set(subset) & set([x,y]))
    if len(intersection) != 1:
        return 0
    return 1
poly_vertices = []
for subset in subsets_list:
    # create vertex
    vertex = [metric(x,y,subset) for x,y,_ in graph.edges()]
    poly_vertices.append(vertex)
    # write vertex to file
    s = ' '.join(str(x) for x in vertex)
    f.write(s + '\n')


# Collect all permutations (1-based for Sage internals)
n_poly_vertices = len(subsets_list)
list_of_all_permutations = []

# Generate symmetries from the graph symmetry group
# Iterate through generators
for gen in gens:
    # Define permutation of vertices by generator
    perm_vertices = gen(vertices)
    # define an empty permutation on the subsets
    perm_subsets_list = []
    # iterate through subsets
    for i, subset in enumerate(subsets_list):
        perm_subset = sorted([perm_vertices[i] for i in subset])
        # Find index of subset in subsets_list
        new_index = -1
        if perm_subset in subsets_list:
            new_index = subsets_list.index(perm_subset)
        else:
            # Take complementary and find that index
            comp_perm_subset = sorted(list(vertices_set.difference(Set(perm_subset))))
            new_index = subsets_list.index(comp_perm_subset)
        if new_index == -1:
            print('no index')
        # 1-based for Sage Permutation
        perm_subsets_list.append(new_index + 1)
    perm = Permutation(perm_subsets_list)
    if perm != Permutation(range(1, n_poly_vertices + 1)):
        list_of_all_permutations.append(perm)


# Define symmetries by subgroups to get restricted symmetry group
# iterate through subsets
for i, vertex_subset in enumerate(subsets_list):
    # set that belongs to the vertex
    vertex_subset = Set(vertex_subset)
    perm_list = []
    for j, permutation_subset in enumerate(subsets_list):
        # set that is used to generate the permutation
        permutation_subset = Set(permutation_subset)
        # calculate symmetric difference of the two sets (and sort)
        symm_diff = permutation_subset.symmetric_difference(vertex_subset)
        symm_diff = sorted(list(symm_diff))
        # find the vertex that is defined by this new set
        if not symm_diff in subsets_list:
            # get the complement of the symmetric difference
            symm_diff = sorted(list(vertices_set.difference(Set(symm_diff))))
        vertex_idx = subsets_list.index(list(symm_diff))
        # 1-based for Sage Permutation
        perm_list.append(vertex_idx + 1)
    perm = Permutation(perm_list)
    if perm != Permutation(range(1, n_poly_vertices + 1)):
        list_of_all_permutations.append(perm)

# Reduce to a small generating set using GAP
G = PermutationGroup([p for p in list_of_all_permutations], domain=range(1, n_poly_vertices + 1))
print("Group order: ", G.order())
gap_G = G._gap_()
gap_gens = gap_G.SmallGeneratingSet()
n_gens = int(gap_gens.Length())
print("Number of generators: ", n_gens)

# Write generators as 0-based permutation lists
f.write('\nVERTEX_PERMUTATIONS:\n')
for k in range(1, n_gens + 1):
    gap_gen = gap_gens[k]
    # Evaluate the permutation on each point (1-based) and convert to 0-based
    perm_list = [int(gap.OnPoints(i, gap_gen)) - 1 for i in range(1, n_poly_vertices + 1)]
    f.write(' '.join(str(x) for x in perm_list) + '\n')

# close file
f.close()
print("Wrote ", output_filename)
