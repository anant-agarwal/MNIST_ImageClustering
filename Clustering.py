"""********************************************************************************************************

CS 5786 : MACHINE LEARNING FOR DATA SCIENCE
Competition 1: Classifying Handwritten Digits

Purpose : Cluster given data points into 10 clusters such that each cluster corresponds to one of the digits from '0' to '9'
Input   : 1) Path of directory containing "features.csv", "Adjacency.csv" and "seed.csv"
Output  : 1) Two csv files containing Id, Category for given data points using best two methods
		  2) Accuracy of both methods on seeds (optional - uncomment to print)

*********************************************************************************************************"""

# Importing relevant packages
import sys

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.manifold import spectral_embedding
from sklearn.mixture import GMM

# Function to compute accuracy of algorithm on seeds
def accuracy(labels, seeds):
    count = 0
    for i in xrange(10):
        for j in xrange(3):
            idx = int(seeds[i, j])
            predicted_label = labels[idx - 1]
            print predicted_label,
            if (predicted_label == i):
                count += 1
        print ""

    print "Accuracy = %d" % count

# Function to get centroids using seeds for 10 clusters
def getCentroids(X_cca, adj_embedding_cca, seeds):
    N, X_d = np.shape(X_cca)
    N, adj_d = np.shape(adj_embedding_cca)

    # Initialize centroids using seed data
    centroids_X_cca = np.zeros((10, X_d))
    centroids_adj_cca = np.zeros((10, adj_d))

    for i in range(10):
        centroids_X_cca[i] = (X_cca[int(seeds[i, 0] - 1)] + X_cca[int(seeds[i, 1] - 1)] +
                              X_cca[int(seeds[i, 2] - 1)]) / 3
        centroids_adj_cca[i] = (adj_embedding_cca[int(seeds[i, 0] - 1)] + adj_embedding_cca[int(seeds[i, 1] - 1)] +
                                adj_embedding_cca[int(seeds[i, 2] - 1)]) / 3

    return centroids_X_cca, centroids_adj_cca

# Function to get centroids using seeds for 30 clusters
def getNewCentroids(X_cca, adj_embedding_cca, seeds):
    N, X_d = np.shape(X_cca)
    N, adj_d = np.shape(adj_embedding_cca)

    # Initialize centroids using seed data
    centroids_X_cca = np.zeros((30, X_d))
    centroids_adj_cca = np.zeros((30, adj_d))

    k = 0
    for i in xrange(10):
        for j in xrange(3):
            centroids_X_cca[k] = X_cca[int(seeds[i, j] - 1)]
            centroids_adj_cca[k] = adj_embedding_cca[int(seeds[i, j] - 1)]
            k += 1

    return centroids_X_cca, centroids_adj_cca


# Parsing user input

if (len(sys.argv) < 2):
    print "Usage: python Clustering.py <directory path>"
    exit(0)

dirname = sys.argv[1]

print "Reading files."

# Read features file
X = np.genfromtxt(dirname + "/features.csv", delimiter=",")

# Read adjacency file
adj = np.genfromtxt(dirname + "/Adjacency.csv", delimiter=",")

# Read seeds file
seeds = np.genfromtxt(dirname + "/seed.csv", delimiter=",")

# Retain subset of features
X_new = X[:,0:13]

print "Computing spectral embedding from adjacency matrix."

# Get spectral embedding from adjacency file
adj_embedding = spectral_embedding(adj, n_components=500, random_state=42, eigen_tol=1e-8)

print "Performing CCA on spectral embedding and features."

# Perform CCA
cca = CCA(n_components=3, scale=False)
X_cca, adj_embedding_cca = cca.fit_transform(X_new, adj_embedding)

print "Performing K-Means clustering using initialized centroids."

# Perform embedding on seeds and get 10 centroids
centroids_X_cca, centroids_adj_cca = getCentroids(X_cca,adj_embedding_cca,seeds)

# Perform K-means using initialized centroids
kmeans_cca = KMeans(n_clusters=10, init=centroids_X_cca).fit_predict(X_cca)

# Uncomment to see accuracy on seeds
# accuracy(kmeans_cca,seeds)

# Perform embedding on seeds and get 30 centroids
centroids_X_cca_30, centroids_adj_cca_30 = getNewCentroids(X_cca,adj_embedding_cca,seeds)

# Perform K-means using initialized centroids
kmeans_cca_30 = KMeans(n_clusters=30, init=centroids_X_cca_30).fit_predict(X_cca)

# Merge 30 clusters in 10 using seeds
kmeans_cca_30_final = np.array(kmeans_cca_30 / 3)

# Uncomment to see accuracy on seeds
# accuracy(kmeans_cca_30_final,seeds)

print "Writing output to file."

# Write output to file
indices = np.arange(1, 12001)

np.savetxt("KMeans_cca_se500_cca3_x13.csv", zip(indices,kmeans_cca), header="Id,Category", comments='', delimiter = ',', fmt= "%d")

np.savetxt("KMeans_cca_se500_cca3_x13_clusters30.csv", zip(indices,kmeans_cca), header="Id,Category", comments='', delimiter = ',', fmt= "%d")