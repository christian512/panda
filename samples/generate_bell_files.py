"""
Generates a bipartite Bell Polytope for given number of inputs and outputs.

Usage: python generate_bell_files.py <inputs_a> <inputs_b> <outputs_a> <outputs_b>
Example: python generate_bell_files.py 2 2 2 2
"""
import argparse
import os
from itertools import product
import numpy as np


def get_deterministic_behaviors_two_party(inputs_a, inputs_b, outputs_a, outputs_b):
    """ Returns all deterministic behaviors corresponding to inputs/outputs for two parties"""
    dim = len(inputs_a) * len(inputs_b) * len(outputs_a) * len(outputs_b)
    # hidden variables
    lhvs_a = product(outputs_a, repeat=len(inputs_a))
    lhvs_b = product(outputs_b, repeat=len(inputs_b))
    deterministics = []
    for lhv_a, lhv_b in product(lhvs_a, lhvs_b):
            # initialize empty behavior
            counter = 0
            d = np.zeros(dim)
            # iterate over possible outcomes and check whether its defined by the LHV
            for a, b in product(range(len(outputs_a)), range(len(outputs_b))):
                for x, y in product(range(len(inputs_a)), range(len(inputs_b))):
                    if lhv_a[x] == a and lhv_b[y] == b:
                        d[counter] = 1.0
                    counter += 1
            deterministics.append(d)
    # check length
    assert len(deterministics) == (len(outputs_a) ** len(inputs_a)) * (len(outputs_b) ** len(inputs_b)), deterministics
    return np.array(deterministics)

def get_relabelling_generators(inputs_a, inputs_b, outputs_a, outputs_b):
    """ Returns a set of relabelling generators for the symmetry group """
    generators = []
    # setup configurations
    configurations = [(a, b, x, y) for a, b, x, y in product(outputs_a, outputs_b, inputs_a, inputs_b)]
    # set the first inputs
    first_x, first_y = inputs_a[0], inputs_b[0]
    # set first outputs
    first_a, first_b = outputs_a[0], outputs_b[0]
    # relabel each input x to the first input
    for curr_x in inputs_a:
        if curr_x == first_x:
            continue
        perm = list(range(len(configurations)))
        for a, b, x, y in configurations:
            if x == curr_x:
                # configuration of current setting
                idx_old = configurations.index((a, b, x, y))
                # same configuration where x is first_x
                idx_new = configurations.index((a, b, first_x, y))
                perm[idx_old] = idx_new
                perm[idx_new] = idx_old
        generators.append(perm)
    # relabel each input y to the first input
    for curr_y in inputs_b:
        if curr_y == first_y:
            continue
        perm = list(range(len(configurations)))
        for a, b, x, y in configurations:
            if y == curr_y:
                # configuration of current setting
                idx_old = configurations.index((a, b, x, y))
                # configuration where y is first_y
                idx_new = configurations.index((a, b, x, first_y))
                perm[idx_old] = idx_new
                perm[idx_new] = idx_old
        generators.append(perm)
    # relabel the first output of first_x with every other output
    for curr_a in outputs_a:
        if curr_a == first_a:
            continue
        perm = list(range(len(configurations)))
        for a, b, x, y in configurations:
            if x == first_x and a == curr_a:
                # current index
                idx_old = configurations.index((a, b, x, y))
                # idx where a is replaced with the first index
                idx_new = configurations.index((first_a, b, x, y))
                # change perm
                perm[idx_old] = idx_new
                perm[idx_new] = idx_old
        generators.append(perm)
    # relabel the first output of first_y with every other output
    for curr_b in outputs_b:
        if curr_b == first_b:
            continue
        perm = list(range(len(configurations)))
        for a, b, x, y in configurations:
            if y == first_y and b == curr_b:
                # current index
                idx_old = configurations.index((a, b, x, y))
                # idx where b is replaced with the first output
                idx_new = configurations.index((a, first_b, x, y))
                # change perm
                perm[idx_old] = idx_new
                perm[idx_new] = idx_old
        generators.append(perm)
    # exchange of parties if number of inputs and outputs is equal on both sides
    if len(inputs_a) == len(inputs_b) and len(outputs_a) == len(outputs_b):
        perm = list(range(len(configurations)))
        for a, b, x, y in configurations:
            idx_old = configurations.index((a, b, x, y))
            idx_new = configurations.index((b, a, y, x))
            perm[idx_old] = idx_new
            perm[idx_new] = idx_old
        generators.append(perm)
    return np.array(generators, dtype=int)

def write_panda_file(vertices, relabeling_generators, filename: str = 'bell'):
    """ Writes Vertices and Maps in PANDA input format.

    PANDA Maps are permutations on coordinate positions expressed via variable names.
    Each line in the Maps section has one token per coordinate: the i-th token is the
    name of the variable that coordinate i maps to.
    E.g. for names [p0, p1, p2], the line "p2 p0 p1" means p0->p2, p1->p0, p2->p1.
    """
    dim = vertices.shape[1]
    names = ['p' + str(i) for i in range(dim)]
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('Names:\n')
        f.write(' '.join(names) + '\n')
        f.write('\n')
        f.write('Maps:\n')
        for perm in relabeling_generators:
            # perm[i] = j means coordinate i is mapped to coordinate j,
            # so the image of variable i is variable perm[i].
            map_line = ' '.join(names[perm[i]] for i in range(dim))
            f.write(map_line + '\n')
        f.write('\n')
        f.write('Vertices:\n')
        for vertex in vertices:
            s = ' '.join(str(int(x)) for x in vertex)
            f.write(s + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a bipartite Bell Polytope in PANDA format.')
    parser.add_argument('inputs_a', type=int, help='Number of inputs for party A')
    parser.add_argument('inputs_b', type=int, help='Number of inputs for party B')
    parser.add_argument('outputs_a', type=int, help='Number of outputs for party A')
    parser.add_argument('outputs_b', type=int, help='Number of outputs for party B')
    args = parser.parse_args()

    inputs_a = list(range(args.inputs_a))
    inputs_b = list(range(args.inputs_b))
    outputs_a = list(range(args.outputs_a))
    outputs_b = list(range(args.outputs_b))

    basename = '{}{}{}{}'.format(args.inputs_a, args.inputs_b, args.outputs_a, args.outputs_b)
    outdir = os.path.join(os.path.dirname(__file__), 'panda_format', 'bell')
    os.makedirs(outdir, exist_ok=True)
    filepath = os.path.join(outdir, basename)

    vertices = get_deterministic_behaviors_two_party(inputs_a, inputs_b, outputs_a, outputs_b)
    vertices = vertices[np.lexsort(np.rot90(vertices))]
    relabeling_generators = get_relabelling_generators(inputs_a, inputs_b, outputs_a, outputs_b)

    write_panda_file(vertices, relabeling_generators, filepath)
    print('Wrote PANDA file: ' + filepath)