import hdbscan
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing





##################################################################
# Exemple : HDBSCAN Clustering


path = './artificial/'
name="cluto-t5-8k.arff"

#path_out = './fig/'
databrut = arff.loadarff(open(path+str(name), 'r'))
datanp = np.array([[x[0],x[1]] for x in databrut[0]])

# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            "+ str(name))
f0 = datanp[:,0] # tous les élements de la première colonne
f1 = datanp[:,1] # tous les éléments de la deuxième colonne
# plt.figure(figsize=(6, 6))
# plt.scatter(f0, f1, s=8)
# plt.title("Donnees initiales : "+ str(name))
# plt.show()
print("------------------------------------------------------")
# print("Appel HDBSCAN (1) ... ")
tps1 = time.time()
model = hdbscan.HDBSCAN(min_cluster_size=10, gen_min_span_tree=True)
model.fit(datanp)
tps2 = time.time()
labels = model.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print('Number of clusters: %d' % n_clusters)
print('Number of noise points: %d' % n_noise)

plt.scatter(f0, f1, c=labels, s=8)
plt.title("Données après clustering HDBSCAN (1)")
plt.show()








# # Appliquer HDBSCAN
# clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
# cluster_labels = clusterer.fit_predict(datanp)

# # Afficher les résultats
# plt.scatter(datanp[:, 0], datanp[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.7)
# plt.title("HDBSCAN Clustering")
# plt.show()
