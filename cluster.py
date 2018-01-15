from sklearn.datasets import load_iris
import numpy as np


layers = 2
leafCellsNr = int(np.power(4, layers))


class Layer:
    def __init__(self, id, num_cells=0):
        self.idx = id
        self.num_cells = num_cells
        self.cells = []


class Cell:
    def __init__(self, min_vals, max_vals):
        self.min_values = min_vals
        self.max_values = max_vals
        self.parent_cell = None
        self.child_cells = None


def create_leaf_layer(min_vals, max_vals, leavesCellNr):
    cell_size = (max_vals - min_vals) / (int(np.sqrt(leavesCellNr)))
    vec_x, vec_y = np.array([cell_size[0], 0.0]), np.array([0.0, cell_size[1]])

    for leaf_y in range(int(np.sqrt(leavesCellNr))):
        for leaf_x in range( int(np.sqrt(leavesCellNr))):
            cell = Cell(np.sum([min_vals, (leaf_x)*vec_x, leaf_y*vec_y], axis=0), np.sum([min_vals, (leaf_x+1)*vec_x, (1+leaf_y)*vec_y], axis=0))

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










