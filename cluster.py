import numpy as np


layers = 2
leafCellsNr = int(np.power(4, layers))
distributions =['normal', 'normal']


class Layer:
    def __init__(self, idx, mins, maxes, num_cells=0):
        self.idx = idx
        self.num_cells = num_cells
        self.cells = []
        self.max_vals = maxes
        self.min_vals = mins

    def get_clusters(self, data):
        """Loads the dataset and sets a cluster for each sample
                """
        cell_id = []
        for sample in data:
            for idx, c in enumerate(self.cells):
                if np.all(np.logical_and(sample >= c.min_values, sample <= c.max_values)):
                    cell_i_id = idx
            cell_id.append(cell_i_id)
        return cell_id

    def get_upper_clusters(self, child_clusters):
        upper_labels = []
        for s in child_clusters:
            upper_labels.append([c.idx for c in self.cells for child in c.child_cells if s == child.idx])
        return upper_labels

    def get_cell_index(self):
        for idx in xrange(self.num_cells):
            yield idx

    def get_cells_stats(self, data):
        """Loads the dataset and computes means and standard deviations for each leaf-layer cell
        """
        labels = np.array(self.get_clusters(data))
        for c in self.cells:
            data_idxs = np.where(labels==c.idx)
            c.mean_values, c.objects_nr, c.standard_devs = np.nanmean(data[data_idxs, :], axis=1), data_idxs[0].size, np.std(data[data_idxs, :], axis=1)
            # print c.objects_nr, c.mean_values, c.standard_devs
        return labels


class Grid:
    def __init__(self, mins=None, maxes=None, n_layers=None):
        self.max_vals = maxes
        self.min_vals = mins
        self.n_layers = n_layers
        self.leafCellsNr = int(np.power(4, n_layers))
        self.layers = []

    def create_leaf_layer(self):
        """Returns leaf-layer of grid structure by given number of cells and initial min
        and maximum values in each dimension"""

        leaf_layer = Layer(self.layers, self.min_vals, self.max_vals, num_cells=self.leafCellsNr)

        # indicate the number of dimensions
        N_DIM = self.min_vals.shape[0]
        n_grid = int(np.power(self.leafCellsNr, 1. / N_DIM))

        # get cell size in each dimension
        cell_size = (self.max_vals - self.min_vals) / n_grid

        # indicate the number of dimensions
        n_dim = cell_size.shape[0]

        # get all dimensions vector to move across the space
        grid_vecs = np.zeros((n_dim, n_dim))
        np.fill_diagonal(grid_vecs, cell_size)

        vec_x, vec_y = np.array([cell_size[0], 0.0]), np.array([0.0, cell_size[1]])

        # TODO: implement dimensionality invariance
        idx_gen = leaf_layer.get_cell_index()

        for leaf_y in range(n_grid):
            for leaf_x in range(n_grid):
                cell = Cell(idx = next(idx_gen), min_vals=np.sum([self.min_vals, (leaf_x) * grid_vecs[0], leaf_y * vec_y], axis=0),
                            max_vals=np.sum([self.min_vals, (leaf_x + 1) * vec_x, (1 + leaf_y) * grid_vecs[1]], axis=0))
                leaf_layer.cells.append(cell)
        self.layers.append(leaf_layer)

    def create_layer(self):
        child_layer = self.layers[-1]
        layer = Layer(len(self.layers),self.min_vals, self.max_vals, child_layer.num_cells/4)
        new_grid = get_upper_grid(child_layer.cells)
        idx_gen = layer.get_cell_index()
        for new_cell_grid in new_grid:
            new_cell = Cell(idx =next(idx_gen), min_vals=new_cell_grid[0].min_values, max_vals=new_cell_grid[-1].max_values)
            new_cell.child_cells = new_cell_grid

            new_cell.objects_nr = np.sum(np.array([c.objects_nr for c in new_cell_grid]))
            new_cell.mean_values = np.nanmean(np.array([c.mean_values for c in new_cell_grid]), axis=0)
            new_cell.calculate_upper_mean_and_std()
            layer.cells.append(new_cell)
            # print new_cell.min_values, new_cell.max_values, new_cell.standard_devs, new_cell.mean_values
        self.layers.append(layer)


def get_upper_grid(li):
    grid = np.array(li)
    pairs = grid.reshape(int(np.sqrt(grid.size)), int(np.sqrt(grid.size)))
    grid = []
    for j in range(0, pairs.shape[1], 2):
        new_col = pairs[[j,j+1],:].reshape((2, -1, 2))
        for pair_nr in range(new_col.shape[1]):
            grid.append(new_col[:,pair_nr,:].ravel())
    return grid


class Cell:
    def __init__(self, idx,  min_vals, max_vals, distributions=None):
        self.min_values = min_vals
        self.max_values = max_vals
        self.mean_values = None
        self.objects_nr = None
        self.standard_devs = None
        self.distributions = distributions
        self.child_cells = None
        self.idx = idx

    def calculate_upper_mean_and_std(self):
        means = np.array([c.mean_values for c in self.child_cells ])
        stdevs = np.array([c.standard_devs for c in self.child_cells])
        obj_counts = np.array([c.objects_nr for c in self.child_cells])
        mean = np.nansum(np.multiply(means.reshape(4,-1), obj_counts.reshape(4,-1)), axis=0)
        self.mean_values = mean/self.objects_nr
        stdev = np.multiply(np.nansum([np.square(means), np.square(stdevs)], axis=0).reshape(4, -1), obj_counts.reshape(4,-1))
        self.standard_devs = np.sqrt((np.sum(stdev, axis=0)/self.objects_nr) - np.square(self.mean_values))


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

def create_grid(data, layers):
    labels = []
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)
    grid = Grid(min_values, max_values, layers)
    grid.create_leaf_layer()
    labels.append(grid.layers[0].get_cells_stats(data))
    for i in range(layers):
        grid.create_layer()
        l = grid.layers[-1]
        labels.append(np.array(l.get_upper_clusters(labels[-1])).ravel())
    return grid, labels






