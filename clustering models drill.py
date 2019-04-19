#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:11:58 2019
clustering models drill
@author: jack
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AffinityPropagation
from itertools import cycle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# The coordinates of the centers of our blobs.
#centers = [[2, 2], [-2, -2], [2, -2],[-3,-3]]
#centers = [[1, 1], [-4, -4], [4, -4]]



def do_work (blob_cnt, r_state, clustrstd, centers):
# Make 10,000 rows worth of data with two features representing three
# clusters, each having a standard deviation of 1.
    X, y = make_blobs(
        n_samples=500*blob_cnt,
        centers=centers,
        cluster_std=clustrstd,
        n_features=2,
        random_state=r_state)
    title = 'base - blobs->' + str(blob_cnt) + ' clutr STD->' + str( clustrstd)
    plt.title(title)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

#Divide into training and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.9,
            random_state=42)
    # Normalize the data.
    X_norm = normalize(X)

# Reduce it to two components.
    X_pca = PCA(2).fit_transform(X_norm)

# Calculate predicted values.
    y_pred = KMeans(n_clusters=blob_cnt, random_state=r_state).fit_predict(X_pca)

# Plot the solution.
    title = 'Kmeans shift - blobs->' + str(blob_cnt) + ' clutr STD->' + str( clustrstd)
    plt.title(title)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred)
    plt.show()


# Here we set the bandwidth. This function automatically derives a bandwidth
# number based on an inspection of the distances among points in the data.
    bandwidth = estimate_bandwidth(X_train, quantile=0.2, n_samples=500)

# Declare and fit the model.
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X_train)

# Extract cluster assignments for each data point.
    labels = ms.labels_

# Count our clusters.
    n_clusters_ = len(np.unique(labels))

#print("Number of estimated clusters: {}".format(n_clusters_))
    title = 'mean shift - blobs->' + str(blob_cnt) + ' clutr STD->' + str( clustrstd)
    plt.title(title)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=labels)
    plt.show()

#print('Comparing the assigned categories to the ones in the data:')
    print("mean shift->",pd.crosstab(y_train,labels))


# We know we're looking for three clusters.
    n_clusters=3

# Declare and fit the model.
    sc = SpectralClustering(n_clusters=n_clusters)
    sc.fit(X_train)

#Predicted clusters.
    predict=sc.fit_predict(X_train)

#Graph results.
    title = 'spectral shift - blobs->' + str(blob_cnt) + ' clutr STD->' + str( clustrstd)
    plt.title(title)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=predict)
    plt.show()

#print('Comparing the assigned categories to the ones in the data:')
    print(pd.crosstab(y_train,predict))

# Declare the model and fit it in one statement.
# Note that you can provide arguments to the model, but we didn't.
    af = AffinityPropagation().fit(X_train)
#print('Done')

# Pull the number of clusters and cluster assignments for each data point.
    cluster_centers_indices = af.cluster_centers_indices_
    n_clusters_ = len(cluster_centers_indices)
    labels = af.labels_

    plt.title('Affinity - Est # of clusters: {}'.format(n_clusters_))
#print('Estimated number of clusters: {}'.format(n_clusters_))

    plt.figure(1)
    plt.clf()

# Cycle through each cluster and graph them with a center point for the
# exemplar and lines from the exemplar to each data point in the cluster.
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = X_train[cluster_centers_indices[k]]
        plt.plot(X_train[class_members, 0], X_train[class_members, 1], col + '.')
        plt.plot(cluster_center[0],
                 cluster_center[1],
                 'o',
                 markerfacecolor=col,
                 markeredgecolor='k')
        for x in X_train[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)
    title = 'affinity - blobs->' + str(blob_cnt) + ' clutr STD->' + str( clustrstd)
    plt.title(title)
    plt.show()

blob_cnt=6
r_state=40
clustrstd=1
centers = [[2, 2], [-2, -2], [2, -2],[-3,-3]]
do_work(blob_cnt, r_state, clustrstd, centers)

blob_cnt=9
r_state=60
clustrstd=2
centers = [[1, 1], [-4, -4], [4, -4]]
do_work(blob_cnt, r_state, clustrstd, centers)

blob_cnt=12
r_state=10
clustrstd=3
centers = [[-3, -3], [-4, -4], [3,3]]
do_work(blob_cnt, r_state, clustrstd, centers)
