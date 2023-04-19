import numpy as np
from matplotlib import pyplot as plt

class Cell:
    """
    Class representing a cell of a regular CW complex
    """
    
    def __init__(self, dim, boundary=None, display_name=None):
        """
        dim is the dimension of the cell
        boundary is a list of cells of dimension dim-1
        """
        if boundary is None:
            boundary = []
        
        assert dim >= 0
        for cell in boundary:
            assert cell.dim == dim-1

        self.dim = dim
        self.boundary = boundary
        self.display_name = display_name
    
def gradient_path_rank(path):
    rank = 0
    for i in range(len(path)-1):
        if path[i].dim > path[i+1].dim:
            rank += path[i].dim - path[i+1].dim - 1
    return rank
    
def path_string(path):
    s = path[0].display_name
    for i in range(1, len(path)):
        if path[i-1].dim > path[i].dim:
            s += " > "
        else:
            s += " < "
        s += path[i].display_name
    return s

class GradientPath(Cell):
    """
    Class representing a morphism in the discrete flow category
    """
    
    def __init__(self, path, boundary=None):
        self.path = path
        dim = gradient_path_rank(path)
        display_name = path_string(path)
        super().__init__(dim, boundary, display_name)

class CWComplex:
    def __init__(self, cells):
        max_dim = max([c.dim for c in cells])
        self.cells = []
        for i in range(max_dim+1):
            self.cells.append([])
        for cell in cells:
            self.cells[cell.dim].append(cell)

    def plot_lattice(self):
        # TODO: can get better ordering: order top layer too so RP2 example gives disjoint component
        # Order the cells to get a nicer visual presentation. Try to minimize edges crossing.
        cells_ordered = []
        max_rank = len(self.cells)-1
        for rank in range(max_rank+1):
            cells_ordered.append([])
        cells_ordered[max_rank] = self.cells[max_rank]
        for rank in range(max_rank, 0, -1):
            cells_unplaced = set([id(c) for c in self.cells[rank-1]])
            for cell in cells_ordered[rank]:
                for face in cell.boundary:
                    if id(face) in cells_unplaced:
                        cells_ordered[rank-1].append(face)
                        cells_unplaced.remove(id(face))
            cells_ordered[rank-1].extend(list(cells_unplaced))
        
        # Create mapping from cells to coordinates (y-axis being rank), and plot points and edges
        coords = {}
        y_coord = np.arange(0, max_rank+1)
        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        for rank in range(max_rank+1):
            x_coord = np.linspace(0, 100, len(cells_ordered[rank])+2)
            for i in range(len(cells_ordered[rank])):
                color = colors[i % len(colors)]
                cell = cells_ordered[rank][i]
                x = x_coord[i+1]
                y = y_coord[rank]
                coords[id(cell)] = [x, y]
                for covered in cell.boundary:
                    x2, y2 = coords[id(covered)]
                    plt.plot([x, x2], [y, y2], color=color)
                plt.plot(x, y, 'k.')
                plt.annotate(cell.display_name, (x, y), rotation=20)
        plt.xticks([])
        plt.yticks([])
        plt.savefig("CW_lattice.pdf")
        plt.show()

def covers(x, y):
    # Returns true if x covers y, i.e., if rho(x)=rho(y)+1 and x > y
    assert x.dim == y.dim + 1
    
    if len(x.path) == len(y.path) + 1:
        longest = x.path
        shortest = y.path
    elif len(x.path) == len(y.path) - 1:
        longest = y.path
        shortest = x.path
    else:
        return False

    diffs = 0
    j = 0
    for i in range(len(longest)):
        if longest[i] != shortest[j]:
            diffs += 1
            if diffs > 1:
                return False
        else:
            j += 1

    return True

def get_boundary(path, Hom_lower_rank):
    bdy = []
    for other in Hom_lower_rank:
        if covers(path, other):
            bdy.append(other)
    return bdy

class FloHomPoset(CWComplex):
    def __init__(self, paths):
        super().__init__(paths)
        self.set_boundaries()
    
    def set_boundaries(self):
        for rank in range(1, len(self.cells)):
            # Rank 0 cells have empty boundary
            for path in self.cells[rank]:
                path.boundary = get_boundary(path, self.cells[rank-1])

def flatten(l):
    return [item for sublist in l for item in sublist]

def get_directed_graph(cells, mu, source, target):
    # Get the directed graph where v -> w if w < v or mu(v) = w 
    
    cell_idx = {}
    for i in range(len(cells)):
        cell_idx[id(cells[i])] = i
    
    # Later optimization: ensure only the elements in the interval [target, source] are in the graph
    graph = [] # Adjacency list representation (but with sets)
    
    # First, add all faces in boundary, and their faces, to the adj. list of each cell
    for i in range(len(cells)):
        graph.append(set())
        for face in cells[i].boundary:
            face_idx = cell_idx[id(face)]
            graph[i].add(face_idx)
            graph[i].update(graph[face_idx])
    
    # Then, add mu of the cell, if it exists
    for i in range(len(cells)):
        mu_i = mu.get(id(cells[i]))
        if mu_i is not None:
            graph[i].add(cell_idx[id(mu_i)])

    return graph, cell_idx[id(source)], cell_idx[id(target)]

def get_simple_paths(graph, source, target, cur_path=None):
    # Traverse the graph to get all paths from source to target that doesn't repeat vertices
    if cur_path == None:
        cur_path = [source]
    elif cur_path[-1] == target:
        return [list(cur_path)]
    
    visited = set(cur_path)
    paths = []
    for cell in graph[cur_path[-1]].difference(visited):
        cur_path.append(cell)
        paths.extend(get_simple_paths(graph, cur_path[-1], target, cur_path))
        cur_path.pop()
    
    return paths

def compute_Flo_Hom(X, mu, source, target):
    # TODO: translating between indices and Cell objects is kind of awkward. More elegent way to do this?
    cells = flatten(X.cells) # Important that this is ordered by rank
    graph, source_idx, target_idx = get_directed_graph(cells, mu, source, target)
    simple_paths = get_simple_paths(graph, source_idx, target_idx)
    paths = [[cells[i] for i in path] for path in simple_paths]
    grad_paths = [GradientPath(path) for path in paths]
    return FloHomPoset(grad_paths)
