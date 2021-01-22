
#load libraries
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

#load data
iris = datasets.load_iris()
features = iris.data

#standardize features
scaler = StandardScaler()
features_std = scaler.fit_transform(features)

#create k-means object
cluster = MiniBatchKMeans(n_clusters=3, random_state=0, batch_size=100)

#train model
model = cluster.fit(features_std)



