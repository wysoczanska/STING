from cluster import *
from sklearn.datasets import load_iris, load_wine
import plot as pt
from sklearn.metrics import silhouette_score, completeness_score, homogeneity_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import csv


n_dim = 2 # okreslenie liczby atrybutow
n_layers = 4 # okreslenie liczby warstw struktury

# Experiment I - iris dataset
iris = load_iris()
X = iris.data[:, :n_dim]
grid, labels = create_grid(X, n_layers) # utworzona struktura + przyporzadkowane klastry

# wizualizacja wynikow - siatka dla poszczegolnych warstw
pt.plot_grid(X, labels, grid)

scores = []
for i, layer in enumerate(grid.layers[:-1]):
    scores.append([len(grid.layers)-i-1, len(np.unique(labels[i])), silhouette_score(X, labels[i])])
pt.plot_n_clusters(np.array(scores))

# eksperyment przy uzyciu algorytmu kMeans
k_num = len(np.unique(labels[-2]))
k_means = KMeans(k_num, random_state=40).fit(X)

pt.plot_results(X, k_means.labels_)

print ('Silhouette score:')
print ('Kmeans: %0.2f STING: %0.2f' %(silhouette_score(X, k_means.labels_), silhouette_score(X, labels[-2])))

# Experiment II
n_layers = 2
wine = load_wine()

X_wine = wine.data[:,:n_dim]
grid_wine, labels_wine = create_grid(X_wine, n_layers)

# Sprawdzenie statystyk na caej dziedzinie
print wine.feature_names
print grid_wine.layers[-1].cells[0].min_values, grid_wine.layers[-1].cells[0].max_values
print grid_wine.layers[-1].cells[0].mean_values, grid_wine.layers[-1].cells[0].standard_devs
pt.plot_grid(X_wine, labels_wine, grid_wine)

scores = []
for i, layer in enumerate(grid_wine.layers[:-1]):
    scores.append([len(grid_wine.layers)-i-1, len(np.unique(labels_wine[i])), silhouette_score(X_wine, labels_wine[i])])
print scores

# pt.plot_n_clusters(np.array(scores))

k_num = len(np.unique(labels_wine[-2]))
k_means = KMeans(k_num, random_state=40).fit(X_wine)

print ('Silhouette score:')
print ('Kmeans: %0.2f STING: %0.2f' %(silhouette_score(X_wine, k_means.labels_), silhouette_score(X_wine, labels_wine[-2])))
print ('Homogenity score:')
print ('Kmeans: %0.2f STING: %0.2f' %( homogeneity_score(wine.target, k_means.labels_), homogeneity_score(wine.target, labels_wine[-2])))
print ('Completeness score:')
print ('Kmeans: %0.2f STING: %0.2f' %( completeness_score(wine.target, k_means.labels_), completeness_score(wine.target, labels_wine[-2])))
pt.plot_results(X_wine, k_means.labels_)

# eksperyment skalowanie danych
X_wine = StandardScaler().fit_transform(X_wine)
grid_wine, labels_wine = create_grid(X_wine, n_layers)
pt.plot_grid(X_wine, labels_wine, grid_wine)
k_means = KMeans(k_num, random_state=40).fit(X_wine)

print ('Silhouette score:')
print ('Kmeans: %0.2f STING: %0.2f' %(silhouette_score(X_wine, k_means.labels_), silhouette_score(X_wine, labels_wine[-2])))
print ('Homogenity score:')
print ('Kmeans: %0.2f STING: %0.2f' %( homogeneity_score(wine.target, k_means.labels_), homogeneity_score(wine.target, labels_wine[-2])))
print ('Completeness score:')
print ('Kmeans: %0.2f STING: %0.2f' %( completeness_score(wine.target, k_means.labels_), completeness_score(wine.target, labels_wine[-2])))
pt.plot_results(X_wine, k_means.labels_)

# Experiment III - pokemons

# loading the data
with open('Pokemon.csv', 'r') as f:
    reader = csv.DictReader(f, delimiter=',')
    X_poke = [[row['HP'], row['Attack']] for row in reader]
X_poke = np.array(X_poke, dtype=np.float)

n_layers = 4
wine = load_wine()

grid_poke, labels_poke = create_grid(X_poke, n_layers)

# Sprawdzenie statystyk na calej dziedzinie
print grid_poke.layers[-1].cells[0].min_values, grid_poke.layers[-1].cells[0].max_values
print grid_poke.layers[-1].cells[0].mean_values, grid_poke.layers[-1].cells[0].standard_devs
pt.plot_grid(X_poke, labels_poke, grid_poke)

scores = []
for i, layer in enumerate(grid_poke.layers[:-1]):
    scores.append([len(grid_poke.layers)-i-1, len(np.unique(labels_poke[i])), silhouette_score(X_poke, labels_poke[i])])

pt.plot_n_clusters(np.array(scores))

k_num = len(np.unique(labels_poke[-3]))
k_means = KMeans(k_num, random_state=40).fit(X_poke)

print ('Silhouette score:')
print ('Kmeans: %0.2f STING: %0.2f' %(silhouette_score(X_poke, k_means.labels_), silhouette_score(X_poke, labels_poke[-3])))
pt.plot_results(X_poke, k_means.labels_)