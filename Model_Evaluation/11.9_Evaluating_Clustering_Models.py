
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
# Generate feature matrix
features, _ = make_blobs(n_samples = 1000,
                         n_features = 10,
                         centers = 3,
                         cluster_std = 0.5,
                         shuffle = True,
                         random_state = 1)
print(features[1])
# Cluster data using k-means to predict classes
model = KMeans(n_clusters=3, random_state=1).fit(features)

#get pridicted classes
target_predicted = model.labels_
print(target_predicted)
#evaluate model
print(silhouette_score(features, target_predicted))































