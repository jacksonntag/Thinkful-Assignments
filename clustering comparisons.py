#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 08:30:01 2019
Clustering comparisons
@author: jack
"""
import itertools
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans,MeanShift, estimate_bandwidth, SpectralClustering
#from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import AffinityPropagation

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)


pd.set_option('display.max_columns', None)
#data_path = ("https://tf-assets-prod.s3.amazonaws.com/"
  #           "tf-curric/data-science/cleveland-data.csv")
headers = ['type','clstrs','RI Score','AdjRandScore','Silouhette']
df_x = pd.DataFrame(columns=headers)

mets = [None]*4
types=0

LABEL = 0
CLUSTERS = 1
KMEANS = 0
MEANSHFT =  1
SPECTRAL = 2
AFFINITY = 3

RISCORE = 2
ARS = 3
METS1=4
METS2=5
METS3=6
METS4=7

type_lbls = ['KMEANS','MEANSHFT', 'SPECTRAL', 'AFFINITY']

data_path = 'BM 2014.csv'
df = pd.read_csv(data_path).dropna()#,skiprows=[0],axis=1, inplace=True
df.dropna(axis=1, inplace=True)
df.drop(['ctz'], axis=1,inplace =True)
df.drop(['bib'], axis=1,inplace =True)

rows = df.shape[0] - df.shape[0] % 4
df = df.iloc[:rows, :]

# Replace some random string values.
X = df.replace(to_replace='?', value=0)
y = df.replace(to_replace='?', value=0)

X = df.iloc[:, :13]
y = df.iloc[:, 13]
y = np.where(y > 0, 0, 1)
# Normalize
save=X
X_norm = normalize(X)

# Data frame to store features and predicted cluster memberships.
ypred = pd.DataFrame()

# Create the two-feature PCA for graphing purposes.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_norm)

# Split the data into four equally-sized samples. First we break it in half:
X_half1, X_half2, X_pcahalf1, X_pcahalf2 = train_test_split(
    X_norm,
    X_pca,
    test_size=0.5,
    random_state=42)

# Then we halve the halves.
X1, X2, X_pca1, X_pca2 = train_test_split(
    X_half1,
    X_pcahalf1,
    test_size=0.5,
    random_state=42) 

X3, X4, X_pca3, X_pca4 = train_test_split(
    X_half2,
    X_pcahalf2,
    test_size=0.5,
    random_state=42)

def do_work(cluster_test, cluster_cnt):
    
# Pass a list of tuples and a counter that increments each time we go
# through the loop. The tuples are the data to be used by k-means,
# and the PCA-derived features for graphing. We use k-means to fit a
# model to the data, then store the predicted values and the two-feature
# PCA solution in the data frame.
    for counter, data in enumerate([
        (X1, X_pca1),
        (X2, X_pca2),
        (X3, X_pca3),
        (X4, X_pca4)]):
    
        # Put the features into ypred.
        ypred['pca_f1' + '_sample' + str(counter)] = data[1][:, 0]
        ypred['pca_f2' + '_sample' + str(counter)] = data[1][:, 1]
        
        # Generate cluster predictions and store them for clusters 2 to 4.
        for nclust in range(2, 5):
            pred = KMeans(n_clusters=cluster_cnt, random_state=42).fit_predict(data[0])
            ypred['clust' + str(nclust) + '_sample' + str(counter)] = pred
            
 
        # Get predicted clusters.
    if (cluster_test == KMEANS):
        print( "In test - >",cluster_test)
        output[LABEL] = type_lbls[cluster_test]
        output[CLUSTERS] = cluster_cnt
        full_pred = KMeans(n_clusters=cluster_cnt, random_state=42).fit_predict(X_norm)
        # Create a list of pairs, where each pair is the ground truth group
        # and the assigned cluster.
        y = df.iloc[:, 13]
        y = np.where(y > 0, 0, 1)
        c = list(itertools.product(y, full_pred))
# Count how often each type of pair (a, b, c, or d) appears.
        #c = np.array(c, dtype=np.float16)
        RIcounts = [[x, c.count(x)] for x in set(c)]
        #print(clusters, " clusters, RIcounts - >",RIcounts)
        # Create the same counts but without the label, for easier math below.
        RIcounts_nolabel = [c.count(x) for x in set(c)]
        # Calculate the Rand Index.
        RIscore = (RIcounts_nolabel[3] + RIcounts_nolabel[2]) / np.sum(RIcounts_nolabel)
        output[RISCORE] = RIscore
        output[ARS] = (metrics.adjusted_rand_score(y, full_pred))
#        print(clusters, " clusters, adjustd Rand Score->", metrics.adjusted_rand_score(y, full_pred))
        output[METS1] = metrics.silhouette_score(X_norm, type_lbls, metric='sqeuclidean')

    if (cluster_test == MEANSHFT):
        #output.clear()
       # output=[0]*8
        print( "In test - >",cluster_test)
        output[LABEL] = type_lbls[cluster_test]
        output[CLUSTERS] = cluster_cnt
        output[RISCORE] = 0#RIscore
        output[ARS] = 0#(metrics.adjusted_rand_score(y, full_pred))
        
        bandwidth = estimate_bandwidth(X_norm, quantile=0.2, n_samples=500)
# Declare and fit the model.
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        full_pred = ms.fit(X_norm)
# Extract cluster assignments for each data point.
        labels = ms.labels_
        y = df.iloc[:, 13]
        y = np.where(y > 0, 0, 1)
        y = print(pd.crosstab(y,labels))
# Coordinates of the cluster centers.
        cluster_centers = ms.cluster_centers_
        output[RISCORE] = 999#RIscore
 ####       output[ARS] = (metrics.adjusted_rand_score(y, full_pred))
        
        output[METS1] = metrics.silhouette_score(X_norm, labels, metric='sqeuclidean')
    # Extract cluster assignments for each data point.
        labels = ms.labels_
    
    if (cluster_test == SPECTRAL):
        print( "In test - >",cluster_test)
        output[LABEL] = type_lbls[cluster_test]
        output[CLUSTERS] = cluster_cnt
        # Declare and fit the model.
        sc = SpectralClustering(n_clusters=cluster_cnt)
        sc.fit(X_norm)
        
        #Predicted clusters.
        y = df.iloc[:, 13]
        y = np.where(y > 0, 0, 1)
        full_pred =sc.fit_predict(X_norm)
        print("spectral->",pd.crosstab(y,full_pred))
        # Create a list of pairs, where each pair is the ground truth group
        # and the assigned cluster.
        c = list(itertools.product(y, full_pred))
        
        # Count how often each type of pair (a, b, c, or d) appears.
        RIcounts = [[x, c.count(x)] for x in set(c)]
        print("Kssspectral  RI Count ->",RIcounts)
        
        output[ARS] = (metrics.adjusted_rand_score(y, full_pred))
        
        # Create the same counts but without the label, for easier math below.
        RIcounts_nolabel = [c.count(x) for x in set(c)]
        # Calculate the Rand Index.
        RIscore = (RIcounts_nolabel[3] + RIcounts_nolabel[2]) / np.sum(RIcounts_nolabel)
        output[RISCORE] = RIscore
#        print("spectral Menas RIscore ->", RIscore)
        output[METS1] = metrics.silhouette_score(X_norm, labels, metric='sqeuclidean')

    if (cluster_test == AFFINITY):
        print( "In test - >",cluster_test)
        output[LABEL] = type_lbls[cluster_test]
        output[CLUSTERS] = cluster_cnt
        
        # Compute Affinity Propagation
        input_x = np.array(X_norm)#  X_norm.values
        af = AffinityPropagation().fit(input_x)
        cluster_centers_indices = af.cluster_centers_indices_
        labels = af.labels_
        print("labels = ",labels)
        n_clusters_ = len(cluster_centers_indices)
        y = df.iloc[:, 13]
        y = np.where(y > 0, 0, 1)
        labels_true = y
        print('Estimated number of clusters: %d' % n_clusters_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f"  \
              % metrics.adjusted_rand_score(labels_true, labels))
        print("Adjusted Mutual Information: %0.3f" \
              % metrics.adjusted_mutual_info_score(labels_true, labels))
 #       print("Silhouette Coefficient: %0.3f" \
 #             % metrics.silhouette_score(X_norm, labels, metric='sqeuclidean'))
        
        output[RISCORE] = 0#RIscore
#        output[ARS] = (metrics.adjusted_rand_score(y, full_pred))
        output[ARS] = metrics.adjusted_rand_score(labels_true, labels)#0#(metrics.adjusted_rand_score(y, full_pred))
        output[METS1] = metrics.silhouette_score(X_norm, labels, metric='sqeuclidean')

#    print("end of do_work->",output)
    
clusters = 10
output = [None]*(len(headers) )
for cluster_cnt in range(3,10,6):
    for test_type in range(KMEANS,AFFINITY+1):
#        print(clusters, test_type)
       do_work(test_type, cluster_cnt)
#       print("before append->",output)
       df_x=df_x.append(pd.DataFrame([output], columns=df_x.columns),ignore_index=True)


print(df_x.head(15))   
