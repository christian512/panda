### Fork Note

This is a fork of the [PANDA repository](https://github.com/stefanloerwald/panda) by Stefan Lörwald.

It aims to improve the perfomance of PANDA using advanced group calculations and recursion.
Additionally, a sampling method, based on the Adjacency Decomposition method, is implemented in this fork. 

### Results
The sampling method was used to enumerate bipartite Bell Polytopes, which were previously not fully enumerated. Due to Github size limitations, the [results of these enumerations can be found elsewhere.](https://nx43005.your-storageshare.de/s/H5z2r8aCsLRrCwm)

### Collins-Gisin parametrization for Bell polytopes
Bipartite Bell polytopes shipped in [samples/panda_format/bell/](samples/panda_format/bell/) use the **Collins-Gisin (CG)** parametrization, a full-dimensional representation obtained by dropping the last outcome of every input. See Collins & Gisin, [arXiv:quant-ph/0306129](https://arxiv.org/abs/quant-ph/0306129) for the original reference.

For a scenario with `m_A` inputs and `o_A` outcomes on side A (and `m_B`, `o_B` on side B), the CG coordinates of a behavior are, in order:

1. A-marginals: `p_A(a|x)` for `a = 0..o_A-2`, `x = 0..m_A-1`  — that is `(o_A-1)·m_A` entries
2. B-marginals: `p_B(b|y)` for `b = 0..o_B-2`, `y = 0..m_B-1`  — that is `(o_B-1)·m_B` entries
3. Joint terms: `p_{AB}(a,b|x,y)` for `a = 0..o_A-2`, `b = 0..o_B-2`, `x = 0..m_A-1`, `y = 0..m_B-1` — that is `(o_A-1)·(o_B-1)·m_A·m_B` entries

The `Names:` block of a Bell sample file reflects this ordering, e.g. for the (2,2,2,2) scenario:

```
Names:
pA0x0 pA0x1 pB0y0 pB0y1 pAB00x0y0 pAB00x0y1 pAB00x1y0 pAB00x1y1
```

The generator [samples/generate_bell_files.py](samples/generate_bell_files.py) produces these files for arbitrary `(m_A, m_B, o_A, o_B)`, including the associated `VERTEX_PERMUTATIONS` (input relabelings, output relabelings, and party exchange when the scenario is symmetric).

### Usage
To use this functionality, you have to manually build PANDA as described in the [install.md](./install.md)

### Official website

The official website for PANDA can be found here:
http://comopt.ifi.uni-heidelberg.de/software/PANDA/


### License

Creative Commons Attribution-NonCommercial 4.0 International License
http://creativecommons.org/licenses/by-nc/4.0/
