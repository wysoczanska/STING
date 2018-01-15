from sklearn.datasets import load_iris
import numpy as np


layers = 2
leafCellsNr = int(np.power(4, layers))
distributions =['normal', 'normal']


class Layer:
    def __init__(self, id, num_cells=0):
        self.idx = id
        self.num_cells = num_cells
        self.cells = []


class Cell:
    def __init__(self, min_vals, max_vals, distributions):
        self.min_values = min_vals
        self.max_values = max_vals
        self.mean_values = None
        self.distributions = distributions
        self.parent_cell = None
        self.child_cells = None


def create_leaf_layer(min_vals, max_vals, leavesCellNr):
    "Returns leaf-layer of grid structure by given number of cells and initial min and maximum values in each dimension"

    # indicate the number of dimensions
    n_dim = len(min_vals)
    n_grid = int(np.power(leavesCellNr, 1./n_dim))

    # get cell size in each dimension
    cell_size = (max_vals - min_vals)/n_grid

    #indicate the number of dimensions
    n_dim = cell_size.shape[0]

    #get all dimensions vector to move across the space
    grid_vecs = np.zeros((n_dim, n_dim))
    np.fill_diagonal(grid_vecs, cell_size)

    vec_x, vec_y = np.array([cell_size[0], 0.0]), np.array([0.0, cell_size[1]])
    print vec_x, vec_y

    #TODO: implement dimensionality invariance
    for leaf_y in range(n_grid):
        for leaf_x in range(n_grid):
            cell = Cell(min_vals=np.sum([min_vals, (leaf_x)*grid_vecs[0], leaf_y*vec_y], axis=0),
                        max_vals=np.sum([min_vals, (leaf_x+1)*vec_x, (1+leaf_y)*grid_vecs[1]], axis=0),
                        distributions=distributions)

            print cell.min_values, cell.max_values


def create_grid(data, leavesCellNr):
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)
    upper_layer = Layer(0)
    upper_layer_cell = Cell(min_values, max_values)
    upper_layer.cells.append(upper_layer_cell)

    structure=[]
    print max_values, min_values
    create_leaf_layer(min_values, max_values, leafCellsNr)

    # for layer_idx in reversed(range(1, layers)):
    #     layer = Layer(layer_idx, np.power(4, layer_idx))
    #     for cell in range(1, layer.num_cells):


iris = load_iris()
X = iris.data[:, :2]
create_grid(X, leafCellsNr)










