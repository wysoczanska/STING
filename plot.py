from matplotlib import pyplot as plt
import numpy as np
from cluster import Grid, Layer, Cell
from itertools import cycle, islice


def plot_grid(data, labels, grid = None):
    for idx, layer in enumerate(grid.layers):
        label = labels[idx]
        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00',
                                             '#FFEBCD', '#006400', '#D2691E',
                                             '#9932CC', '#E9967A', '#CD5C5C',
                                             '#ADFF2F', '#FFD700', '#FFD890']),
                                      int(max(label) + 11))))
        fig = plt.figure(idx)
        ax = fig.gca()
        xs = [c.min_values[0] for c in layer.cells]
        xs.append(layer.cells[-1].max_values[0])
        ys = [c.min_values[1] for c in layer.cells]
        ys.append(layer.cells[-1].max_values[1])
        ax.set_xticks(np.unique(np.array(xs)))
        ax.set_yticks(np.unique(np.array(ys)))
        plt.scatter(data[:,0], data[:,1], color = colors[label])
        plt.title('Grid layer %i' %idx)
        plt.grid()
        plt.show()

def plot_results(data, labels):
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00',
                                         '#FFEBCD', '#006400', '#D2691E',
                                         '#9932CC', '#E9967A', '#CD5C5C',
                                         '#ADFF2F', '#FFD700']),
                                  int(max(labels) + 2))))
    fig = plt.figure()
    ax = fig.gca()
    plt.xticks(())
    plt.yticks(())
    plt.scatter(data[:, 0], data[:, 1], color=colors[labels])
    plt.title('Kmeans algorithm k = %i' %len(np.unique(labels)))
    plt.grid()
    plt.show()


def plot_n_clusters(data):

    fig = plt.figure()
    plt.yticks(np.asarray(data[:,0], dtype=int))
    plt.ylabel('Layer id')
    plt.xlabel('Number of clusters')
    plt.scatter(data[:,1], data[:,0])
    plt.plot(data[:,1], data[:,0],'r--', label = 'Number of real clusters')
    y2 = np.power(4, data[:,0])
    print y2
    plt.plot(y2, data[:,0], 'g', label ='Number of cells')
    plt.title('Real clusters number in each grid layer')
    plt.legend()
    plt.show()


