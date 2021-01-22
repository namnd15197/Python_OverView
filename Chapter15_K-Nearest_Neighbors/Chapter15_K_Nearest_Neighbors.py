#Finding an Observation's Nearest Neighbors
#load libraries
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

#load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

#create standardScaler
standardscaler = StandardScaler()

#Standardize features
features_standardized = standardscaler.fit_transform(features)

#two nearest neighbors
nearest_neighbors = NearestNeighbors(n_neighbors=2).fit(features_standardized)

#create an observation
new_observation = [1, 1, 1, 1]
#find distance and indices of the observation's nearest neighbors
distances, indices = nearest_neighbors.kneighbors([new_observation])
# View distances
print("Distance Minkowski: ", distances)
#view the nearest neighbors
print(features_standardized[indices])

# Find two nearest neighbors based on euclidean distance
nearestneighbors_euclidean = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(features_standardized)

distances, indices = nearestneighbors_euclidean.kneighbors([new_observation])
# View distances
print("Distance Euclidean: n_neighbors=2", distances)

# Find each observation's three nearest neighbors
# based on euclidean distance (including itself)
nearestneighbors_euclidean = NearestNeighbors(n_neighbors=3, metric="euclidean").fit(features_standardized)
distances, indices = nearestneighbors_euclidean.kneighbors([new_observation])
# View distances
print("Distance Euclidean: n_neighbors=3", distances)


# List of lists indicating each observation's 3 nearest neighbors
# (including itself)
nearest_neighbors_with_self = nearestneighbors_euclidean.kneighbors_graph( features_standardized).toarray()
# Remove 1's marking an observation is a nearest neighbor to itself
for i, x in enumerate(nearest_neighbors_with_self):
 x[i] = 0
# View first observation's two nearest neighbors
print("nearest neighbors with self: ", nearest_neighbors_with_self[0])

