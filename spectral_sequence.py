import numpy as np
from matplotlib import pyplot as plt

from discrete_flow import compute_Flo_Hom, GradientPath
from utils import flatten

"""
Note: Here I (sloppily) use the term "p-morphism" to mean a set of morphisms {f_0, ..., f_p}
with f_0 < ... < f_p. This could more correctly be called a non-degenerate p-simplex
in S_{*,1}, where S is the double nerve of the discrete flow category, or, equivalently,
an element of E^0_{p,1}.
"""

# TODO: Maybe create p-morphism class?

class FloSpecSeq:
    def __init__(self, X, mu, coeffs="rational"):
        # X is a CWcomplex object
        # mu: U -> D is a discrete Morse function
        assert coeffs in ["rational"] # TODO: add more options for coeffs
        self.coeffs=coeffs
 
        # 1. generate all Hom posets between critical cells
        critical_cells = get_critical(X, mu)
        num_objs = len(critical_cells)
        Hom = []
        for i in range(num_objs):
            c = critical_cells[i]
            for d in critical_cells:
                # TODO: This is inefficient. Better way: compute all paths in graph starting at ending at any critical vertex. Add each one to their respective Hom set
                Hom.append(compute_Flo_Hom(X, mu, c, d))
                # TODO: compute_Flo_Hom doesn't return identity morphisms. It's really what we want here though, but it is unintuitive

        # TODO: remove later
        for poset in Hom:
            poset.plot_lattice()

        # Find indecomposable morphisms
        morphisms = flatten([flatten(h.cells) for h in Hom])
        indecomposables = []
        for m in morphisms:
            if indecomposable(m, critical_cells):
                indecomposables.append(m)

        # construct dict from id(morphism) to index
        morphism_idx = {}
        for i in range(len(indecomposables)):
            hash = p_morphism_hash([indecomposables[i]])
            morphism_idx[hash] = i
    
        # Initialize page 1
        max_dim_X = len(X.cells)-1
        self.E1 = np.zeros((max_dim_X, 2), dtype=np.int32)
        self.E1[0,1] = len(indecomposables) # This is not really E1_01, the real value will be set later

        # 2. Use Hom(c, d) and indecomposables of Hom(c, d)  to compute E1_p1 for p >= 1
        p_morphisms = [[indecomposables]]
        del1_01 = np.zeros((1, len(indecomposables))) # Just a zero map
        del1_x1 = [del1_01]
        p_morphism_idx = [morphism_idx]
        for p in range(1, max_dim_X):
            # Find length p paths in each Hom set that starts with an indecomposable
            p_morphisms.append(get_p_morphisms(Hom, p, critical_cells))
            self.E1[p, 1] = len(p_morphisms[p])

            p_morphism_idx.append({})
            del1_p1 = np.zeros((self.E1[p-1,1], self.E1[p,1]))
            for i in range(len(p_morphisms[p])):
                p_morphism = p_morphisms[p][i]
                hash = p_morphism_hash(p_morphism)
                p_morphism_idx[p][hash] = i

                for j in range(p+1):
                    # TODO: refactor to avoid so many nested loops/if statements
                    dj = p_morphism[:j] + p_morphism[j+1:]
                    dj_decomp = decompose(dj, critical_cells)
                    for factor in dj_decomp:
                        factor_hash = p_morphism_hash(factor)
                        factor_idx = p_morphism_idx[p-1].get(factor_hash)
                        if factor_idx is not None:
                            # factor_idx is None when a factor is degenerate
                            del1_p1[factor_idx,i] = (-1)**j

            del1_x1.append(del1_p1)

        # 3. Compute \partial^0_01 to get E1_01 and E1_00
        del0_01 = np.zeros((num_objs, self.E1[0,1]))
        critical_cell_idx = {}
        for i in range(len(critical_cells)):
            c = critical_cells[i]
            critical_cell_idx[id(c)] = i
        for i in range(self.E1[0,1]):
            f = indecomposables[i]
            source = f.path[0]
            source_idx = critical_cell_idx[id(source)]
            target = f.path[-1]
            target_idx = critical_cell_idx[id(target)]
            del0_01[target_idx, i] = 1
            del0_01[source_idx, i] = -1
        self.E1[0,0] = num_objs - np.linalg.matrix_rank(del0_01)
        self.E1[0,1] = self.E1[0,1] - np.linalg.matrix_rank(del0_01)

        # 4. Compute \partial^1_p1 for p >= 1 to get E2
        self.E2 = np.array(self.E1)
        for p in range(max_dim_X-1):
            self.E2[p,1] = self.E2[p,1] - np.linalg.matrix_rank(del1_x1[p+1])
            self.E2[p+1,1] = self.E2[p+1,1] - np.linalg.matrix_rank(del1_x1[p+1])

    def plot_page1(self):
        print_spec_seq(self.E1, "page 1", self.coeffs)
        return

    def plot_page2(self):
        print_spec_seq(self.E2, "page 2", self.coeffs)
        return

def get_p_morphisms(Hom, p, critical_cells):
    # Returns number of indecomposable p-morphims, i.e., (f_0 < ... < f_p) with f_p indecomposable
    p_morphisms = []
    for poset in Hom:
        for morphism in flatten(poset.cells):
            if indecomposable(morphism, critical_cells):
                p_morphisms.extend(length_n_paths(poset, n=p+1, dest=morphism))
    return p_morphisms

def length_n_paths(poset, n, dest):
    # Returns the paths of the form x_1 < ... < x_n = dest
    if n == 1:
        return [[dest]]
    paths = []
    for cell in dest.faces:
        n_minus_1_paths = length_n_paths(poset, n-1, cell)
        for path in n_minus_1_paths:
            paths.append(path + [dest])
    return paths

def p_morphism_hash(p_morphism):
    l = []
    for morphism in p_morphism:
        l.append(morphism_hash(morphism))
    return tuple(l)

def morphism_hash(morphism):
    l = []
    for cell in morphism.path:
        l.append(id(cell))
    return tuple(l)

def decompose(p_morphism, critical_cells):
    # Return the decomposition of a p-morphism
    critical_cell_ids = set(id(c) for c in critical_cells)
    f_p_decomp = decompose_single(p_morphism[-1], critical_cell_ids)
    obj_ids = {id(f.path[-1]) for f in f_p_decomp} # the objects that f_p factorizes through
    # TODO: it's probably better to just work with lists of cells instead of GradientPath objects
    f_x_decomp = []
    for i in range(len(p_morphism)-1):
        f_i_decomp = decompose_single(p_morphism[i], obj_ids)
        f_x_decomp.append(f_i_decomp)
    f_x_decomp.append(f_p_decomp)
    decomp = []
    for i in range(len(obj_ids)):
        factor_i = [f_j_decomp[i] for f_j_decomp in f_x_decomp]
        decomp.append(factor_i)
    return decomp

def decompose_single(morphism, critical_cell_ids):
    # Returns the decomposition of a single morphism through objs in critical_cell_ids
    decomp = []
    cur_factor = [morphism.path[0]]
    for cell in morphism.path[1:]:
        cur_factor.append(cell)
        if id(cell) in critical_cell_ids:
            decomp.append(GradientPath(cur_factor))
            cur_factor = [cell]
    return decomp

def print_spec_seq(table, title=None, coeffs="rational"):
    if coeffs=="rational":
        symbol = r"\mathbb{Q}"
    else:
        raise "unsupported coefficients"
    for x in range(table.shape[0]):
        for y in range(table.shape[1]):
            if table[x][y] != 0:
                group = "$" + symbol + "^{" + str(table[x][y]) + "}$"
                plt.text(x, y, group)
    xylim = max(table.shape[0], table.shape[1]) - 1
    plt.xlim([-.2, xylim + .2])
    plt.ylim([-.2, xylim + .2])
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show() 

def get_critical(X, mu):
    # X is a CWcomplex object
    # mu: U -> D is a discrete Morse function
    cells = flatten(X.cells)
    critical_cell_ids = set(id(c) for c in cells)
    for key, val in mu.items():
        critical_cell_ids.remove(key)
        critical_cell_ids.remove(id(val))
    critical_cells = [c for c in cells if id(c) in critical_cell_ids]
    return critical_cells

def indecomposable(gradient_path, critical_cells):
    critical_cell_ids = set(id(c) for c in critical_cells)
    for c in gradient_path.path[1:-1]:
        if id(c) in critical_cell_ids:
            return False
    return True
