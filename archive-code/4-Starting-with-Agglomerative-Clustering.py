import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics


###################################################################
# Exemple : Agglomerative Clustering


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


### FIXER la distance
# 
tps1 = time.time()
seuil_dist=10
model = cluster.AgglomerativeClustering(distance_threshold=seuil_dist, linkage='average', n_clusters=None)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# Nb iteration of this method
#iteration = model.n_iter_
k = model.n_clusters_
leaves=model.n_leaves_
plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif (average, distance_treshold= "+str(seuil_dist)+") "+str(name))
plt.show()
print("nb clusters =",k,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")


###
# FIXER le nombre de clusters
###
k=2
tps1 = time.time()
model = cluster.AgglomerativeClustering(linkage='single', n_clusters=k)
model = model.fit(datanp)
tps2 = time.time()
labels = model.labels_
# Nb iteration of this method
#iteration = model.n_iter_
kres = model.n_clusters_
leaves=model.n_leaves_
#print(labels)
#print(kres)

plt.scatter(f0, f1, c=labels, s=8)
plt.title("Clustering agglomératif (single, n_cluster= "+str(k)+") "+str(name))
plt.show()
print("nb clusters =",kres,", nb feuilles = ", leaves, " runtime = ", round((tps2 - tps1)*1000,2),"ms")

# Clustering agglomératif itératif 

silhouette_score = []
ti = time.time()
for i in range(2, 42):
    model = cluster.AgglomerativeClustering(linkage='ward', n_clusters=i)
    model.fit(datanp)
    silhouette_score.append([i, metrics.silhouette_score(datanp, model.labels_)])
tf= time.time()
silhouette_score = np.array(silhouette_score)

print("Total time : ", tf - ti)

#plt.figure(figsize=(6, 6))
plt.title("Evolution du coefficient de silhouette : "+ str(name))
plt.plot(silhouette_score[:, 0],silhouette_score[:, 1])
plt.xticks(range(0, 42))
plt.grid(visible=True)
#plt.savefig(path_out+"Plot-kmeans-code1-"+str(name)+"-cluster.jpg",bbox_inches='tight', pad_inches=0.1)
plt.show()


from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

# Cluster (centroids)
unique_labels = np.unique(labels)
centroids = np.array([np.mean(datanp[labels == label], axis=0) for label in unique_labels])

clusters_distances = pairwise_distances(datanp, centroids)

print("Séparation des centres")
for i, centroid in enumerate(centroids):
	cluster_distances = clusters_distances[labels == unique_labels[i], i]
	print("Cluster ", i, " :")
	print("Min : ", np.min(cluster_distances))
	print("Max : ", np.max(cluster_distances))
	print("Mean : ", np.mean(cluster_distances))
	print()

dists = euclidean_distances(centroids)

print("Séparation des centres")
print("Min : ", np.min(dists[dists > 0]))
print("Max : ", np.max(dists))
print("Mean : ", np.mean(dists[dists > 0]))
print()