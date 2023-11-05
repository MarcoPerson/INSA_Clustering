"""
Created on 2023/09/11

@author: huguet
"""
import os
os.environ["OMP_NUM_THREADS"] = '4'

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics

##################################################################
# Exemple :  k-Means Clustering

path = './artificial/'
name="banana.arff"

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

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, s=8)
plt.title("Donnees initiales : "+ str(name))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-init.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

# Run clustering method for a given number of clusters
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée")
tps1 = time.time()
k=2
model = cluster.MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=1)
model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# informations sur le clustering obtenu
iteration = model.n_iter_
inertie = model.inertia_
centroids = model.cluster_centers_

#plt.figure(figsize=(6, 6))
plt.scatter(f0, f1, c=labels, s=8)
plt.scatter(centroids[:, 0],centroids[:, 1], marker="x", s=50, linewidths=3, color="red")
plt.title("Données après clustering : "+ str(name) + " - Nb clusters ="+ str(k))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

print("nb clusters =",k,", nb iter =",iteration, ", inertie = ",inertie, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
#print("labels", labels)

from sklearn.metrics.pairwise import euclidean_distances
dists = euclidean_distances(centroids)
#print(dists)

"""
# 2-1
# les scores de regroupement de chaque cluster
from sklearn.metrics.pairwise import pairwise_distances
clusters_distances = pairwise_distances(datanp, centroids)


# les scores de regroupement entre les clusters
for i in range(k):
    cluster_distances = clusters_distances[labels == i, i]
    print("Cluster ", i, " :")
    print("Min : ", np.min(cluster_distances))
    print("Max : ", np.max(cluster_distances))
    print("Mean : ", np.mean(cluster_distances))
    print()

# les scores de séparation entre les clusters
print("Séparation des centres")
print("Min : ", np.min(dists[dists>0]))
print("Max : ", np.max(dists))
print("Mean : ", np.mean(dists[dists>0]))
print()"""

# 2-1 et 2-2
"""inertieTab = []
for i in range(2, 42):
    model = cluster.KMeans(n_clusters=i, init='k-means++', n_init=1)
    model.fit(datanp)
    inertieTab.append([i, model.inertia_])
inertieTab = np.array(inertieTab)
print(inertieTab)

#plt.figure(figsize=(6, 6))
plt.plot(inertieTab[:, 0],inertieTab[:, 1])
plt.xticks(range(0, 42))
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()"""

# 2-3
silhouette_score = []
ti = time.time()
for i in range(2, 42):
    model = cluster.MiniBatchKMeans(n_clusters=i, init='k-means++', n_init=1)
    model.fit(datanp)
    silhouette_score.append([i, metrics.silhouette_score(datanp, model.labels_)])
tf= time.time()
silhouette_score = np.array(silhouette_score)

print("Total time : ", tf - ti)

#plt.figure(figsize=(6, 6))

plt.plot(silhouette_score[:, 0],silhouette_score[:, 1])
plt.xticks(range(0, 42))
plt.grid(visible=True)
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()

### Le score de regroupement et de separation ne change pas étant donné que la meileure repartition reste pour k=3

# 2-4
# Chainlink Banana 3-Spiral
# R15 2d-4c-no9 2d-4c

# 2-5 
# Nothing changed except the time reduced




