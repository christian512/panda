"""
Generates a bipartite Bell Polytope for given number of inputs and outputs,
using the full probability parameterization.

The full probability coordinates for a behavior are:
  - p(a,b|x,y) for a=0..o_A-1, b=0..o_B-1, x=0..m_A-1, y=0..m_B-1

Unlike the Collins-Gisin parameterization (see generate_bell_files.py), no
outcome is dropped, so the representation is not full-dimensional: the
normalization and no-signaling conditions hold as implicit equations.

For deterministic behaviors all entries are 0 or 1.

Usage: python generate_bell_files_full_probability.py <inputs_a> <inputs_b> <outputs_a> <outputs_b>
Example: python generate_bell_files_full_probability.py 2 2 2 2
"""
import argparse
import os
from itertools import product
import numpy as np


def fp_vector(lhv_a, lhv_b, m_A, m_B, o_A, o_B):
    """Compute full probability coordinates for a deterministic behavior (lhv_a, lhv_b)."""
    components = []
    for a in range(o_A):
        for b in range(o_B):
            for x in range(m_A):
                for y in range(m_B):
                    components.append(1 if (lhv_a[x] == a and lhv_b[y] == b) else 0)
    return np.array(components, dtype=int)


def fp_names(m_A, m_B, o_A, o_B):
    """Return coordinate names for the full probability parameterization."""
    names = []
    for a in range(o_A):
        for b in range(o_B):
            for x in range(m_A):
                for y in range(m_B):
                    names.append(f'pAB{a}{b}x{x}y{y}')
    return names


def get_deterministic_behaviors_fp(m_A, m_B, o_A, o_B):
    """Returns (sorted_vectors, strategies, vec_to_idx) in full probability coordinates.

    sorted_vectors: numpy array, one probability vector per row, lexicographically sorted
    strategies:     list of (lhv_a, lhv_b) tuples in the same order
    vec_to_idx:     dict mapping tuple(probability vector) -> sorted index
    """
    strategies = []
    vectors = []
    for lhv_a, lhv_b in product(product(range(o_A), repeat=m_A),
                                 product(range(o_B), repeat=m_B)):
        vectors.append(fp_vector(lhv_a, lhv_b, m_A, m_B, o_A, o_B))
        strategies.append((lhv_a, lhv_b))

    vectors = np.array(vectors, dtype=int)
    sort_idx = np.lexsort(np.rot90(vectors))
    vectors = vectors[sort_idx]
    strategies = [strategies[i] for i in sort_idx]
    vec_to_idx = {tuple(v): i for i, v in enumerate(vectors)}
    return vectors, strategies, vec_to_idx


def get_vertex_permutations_fp(strategies, vec_to_idx, m_A, m_B, o_A, o_B):
    """Returns relabeling generators as permutations on sorted vertex indices.

    Generators:
      - swap input x <-> 0 for party A (m_A-1 generators)
      - swap input y <-> 0 for party B (m_B-1 generators)
      - swap output 0 <-> a at x=0 for party A (o_A-1 generators)
      - swap output 0 <-> b at y=0 for party B (o_B-1 generators)
      - party exchange if m_A==m_B and o_A==o_B (1 generator)
    """
    def lookup(lhv_a, lhv_b):
        return vec_to_idx[tuple(fp_vector(lhv_a, lhv_b, m_A, m_B, o_A, o_B))]

    generators = []

    for x_swap in range(1, m_A):
        perm = []
        for lhv_a, lhv_b in strategies:
            a_new = list(lhv_a)
            a_new[0], a_new[x_swap] = a_new[x_swap], a_new[0]
            perm.append(lookup(tuple(a_new), lhv_b))
        generators.append(perm)

    for y_swap in range(1, m_B):
        perm = []
        for lhv_a, lhv_b in strategies:
            b_new = list(lhv_b)
            b_new[0], b_new[y_swap] = b_new[y_swap], b_new[0]
            perm.append(lookup(lhv_a, tuple(b_new)))
        generators.append(perm)

    for a_swap in range(1, o_A):
        perm = []
        for lhv_a, lhv_b in strategies:
            a_new = list(lhv_a)
            if a_new[0] == 0:
                a_new[0] = a_swap
            elif a_new[0] == a_swap:
                a_new[0] = 0
            perm.append(lookup(tuple(a_new), lhv_b))
        generators.append(perm)

    for b_swap in range(1, o_B):
        perm = []
        for lhv_a, lhv_b in strategies:
            b_new = list(lhv_b)
            if b_new[0] == 0:
                b_new[0] = b_swap
            elif b_new[0] == b_swap:
                b_new[0] = 0
            perm.append(lookup(lhv_a, tuple(b_new)))
        generators.append(perm)

    if m_A == m_B and o_A == o_B:
        perm = [lookup(lhv_b, lhv_a) for lhv_a, lhv_b in strategies]
        generators.append(perm)

    return generators


def write_panda_file(vectors, names, vertex_perms, filename):
    """Writes Names, VERTEX_PERMUTATIONS, and Vertices in PANDA input format."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('Names:\n')
        f.write(' '.join(names) + '\n')
        f.write('\n')
        f.write('Vertices:\n')
        for v in vectors:
            f.write(' '.join(str(int(x)) for x in v) + '\n')
        f.write('\n')
        f.write('VERTEX_PERMUTATIONS:\n')
        for perm in vertex_perms:
            f.write(' '.join(str(i) for i in perm) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a bipartite Bell Polytope in PANDA format (full probability).')
    parser.add_argument('inputs_a', type=int, help='Number of inputs for party A')
    parser.add_argument('inputs_b', type=int, help='Number of inputs for party B')
    parser.add_argument('outputs_a', type=int, help='Number of outputs for party A')
    parser.add_argument('outputs_b', type=int, help='Number of outputs for party B')
    args = parser.parse_args()

    m_A, m_B, o_A, o_B = args.inputs_a, args.inputs_b, args.outputs_a, args.outputs_b

    basename = '{}{}{}{}'.format(m_A, m_B, o_A, o_B)
    outdir = os.path.join(os.path.dirname(__file__), 'panda_format', 'bell_full_probability')
    os.makedirs(outdir, exist_ok=True)
    filepath = os.path.join(outdir, basename)

    vectors, strategies, vec_to_idx = get_deterministic_behaviors_fp(m_A, m_B, o_A, o_B)
    vertex_perms = get_vertex_permutations_fp(strategies, vec_to_idx, m_A, m_B, o_A, o_B)
    names = fp_names(m_A, m_B, o_A, o_B)

    write_panda_file(vectors, names, vertex_perms, filepath)
    print(f'Wrote PANDA file: {filepath}')
    print(f'  Vertices: {len(vectors)}  (full probability dimension: {vectors.shape[1]})')
    print(f'  Symmetry generators: {len(vertex_perms)}')
