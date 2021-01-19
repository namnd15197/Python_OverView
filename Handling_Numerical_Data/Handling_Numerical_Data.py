
#4.1 Rescaling a Feature

# Load libraries
import numpy as np
from sklearn import preprocessing
# Create feature
feature = np.array([[-500.5],
                     [-100.1],
                     [0],
                     [100.1],
                     [900.9]])

# Create scaler
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))

# Scale feature
scaled_feature = minmax_scale.fit_transform(feature)

# Show feature
print(" Scale Feature: \n", scaled_feature)

print("\n")
print("\n")
print("\n")


#4.2 Standardizing a Feature

# Load libraries
import numpy as np
from sklearn import preprocessing
# Create feature
x = np.array([[-1000.1],
 [-200.2],
 [500.5],
 [600.6],
 [9000.9]])
# Create scaler
scaler = preprocessing.StandardScaler()
# Transform the feature
standardized = scaler.fit_transform(x)
# Show feature
print("Standardized: \n", standardized)

print("\n")
print("\n")
print("\n")


# Create scaler
robust_scaler = preprocessing.RobustScaler()
# Transform feature
print("Robust Scaler: \n", robust_scaler.fit_transform(x))
print("\n")
print("\n")
print("\n")

#4.3 Normalizing Observations

# Load libraries
import numpy as np
from sklearn.preprocessing import Normalizer
# Create feature matrix
features = np.array([[0.5, 0.5],
 [1.1, 3.4],
 [1.5, 20.2],
 [1.63, 34.4],
 [10.9, 3.3]])

# Create normalizer
normalizer = Normalizer(norm="l2")
# Transform feature matrix
print("normalizing: \n", normalizer.transform(features))

print("\n")
print("\n")
print("\n")


#4.4 Generating Polynomial and Interaction Features

# Load libraries
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
# Create feature matrix
features = np.array([[2, 3],
                     [2, 3],
                     [2, 3]])    
# Create PolynomialFeatures object
polynomial_interaction = PolynomialFeatures(degree=2, include_bias=False)

# Create polynomial features
print("Polynomial Interaction: \n", polynomial_interaction.fit_transform(features))

print("\n")
print("\n")
print("\n")


#4.5 Transforming Features

# Load libraries
import numpy as np
from sklearn.preprocessing import FunctionTransformer
# Create feature matrix
features = np.array([[2, 3],
 [2, 3],
 [2, 3]])
# Define a simple function
def add_ten(x):
 return x + 10

# Create transformer
ten_transformer = FunctionTransformer(add_ten)
# Transform feature matrix
print("Transform features: \n", ten_transformer.transform(features))

print("\n")
print("\n")
print("\n")


# Load library
import pandas as pd
# Create DataFrame
df = pd.DataFrame(features, columns=["feature_1", "feature_2"])
# Apply function
print(df.apply(add_ten))


print("\n")
print("\n")
print("\n")

#4.6 Detecting Outliers

# Load libraries
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs
# Create simulated data
features, _ = make_blobs(n_samples = 10,
                         n_features = 2,
                         centers = 1,
                         random_state = 1)
# Replace the first observation's values with extreme values
features[0,0] = 10000
features[0,1] = 10000
# Create detector
outlier_detector = EllipticEnvelope(contamination=.1)
# Fit detector
outlier_detector.fit(features)
# Predict outliers
print("Detecting outliers: \n", outlier_detector.predict(features))

# Create one feature
feature = features[:,0]
# Create a function to return index of outliers
def indicies_of_outliers(x):
 q1, q3 = np.percentile(x, [25, 75])
 iqr = q3 - q1
 lower_bound = q1 - (iqr * 1.5)
 upper_bound = q3 + (iqr * 1.5)
 return np.where((x > upper_bound) | (x < lower_bound))
# Run function
print(indicies_of_outliers(feature))

print("\n")
print("\n")
print("\n")

#4.7 Handling Outliers

# Load library
import pandas as pd
# Create DataFrame
houses = pd.DataFrame()
houses['Price'] = [534433, 392333, 293222, 4322032]
houses['Bathrooms'] = [2, 3.5, 2, 116]
houses['Square_Feet'] = [1500, 2500, 1500, 48000]
# Filter observations
print("Filter observations: \n", houses[houses['Bathrooms'] < 20])

print("\n")
print("\n")
print("\n")

# Load library
import numpy as np
# Create feature based on boolean condition
houses["Outlier"] = np.where(houses["Bathrooms"] < 20, 0, 1)
# Show data
print(houses)

# Log feature
houses["Log_Of_Square_Feet"] = [np.log(x) for x in houses["Square_Feet"]]
# Show data
print(houses)


print("\n")
print("\n")
print("\n")


#4.8 Discretizating Features

# Load libraries
import numpy as np
from sklearn.preprocessing import Binarizer
# Create feature
age = np.array([[6],
 [12],
 [20],
 [36],
 [65]])
# Create binarizer
binarizer = Binarizer(18)
# Transform feature
binarizer.fit_transform(age)
# Bin feature
np.digitize(age, bins=[20,30,64])
# Bin feature
np.digitize(age, bins=[20,30,64], right=True)

# Bin feature
np.digitize(age, bins=[18])



#4.9 Grouping Observations Using Clustering
# Load libraries
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
# Make simulated feature matrix
features, _ = make_blobs(n_samples = 50,
 n_features = 2,
 centers = 3,
 random_state = 1)
# Create DataFrame
dataframe = pd.DataFrame(features, columns=["feature_1", "feature_2"])
# Make k-means clusterer
clusterer = KMeans(3, random_state=0)
# Fit clusterer
clusterer.fit(features)
# Predict values
dataframe["group"] = clusterer.predict(features)
# View first few observations
print(dataframe.head(5))

print("\n")
print("\n")
print("\n")

#4.10 Deleting Observations with Missing Values

# Load library
import numpy as np
# Create feature matrix
features = np.array([[1.1, 11.1],
                     [2.2, 22.2],
                     [3.3, 33.3],
                     [4.4, 44.4],
                     [np.nan, 55]])
# Keep only observations that are not (denoted by ~) missing
print("Keep only observations not null: \n", features[~np.isnan(features).any(axis=1)])

# Load library
import pandas as pd
# Load data
dataframe = pd.DataFrame(features, columns=["feature_1", "feature_2"])
# Remove observations with missing values
print(dataframe.dropna())
print("\n")
print("\n")
print("\n")

#4.11 Imputing Missing Values

## Load libraries
#import numpy as np
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
## Make a simulated feature matrix
features, _ = make_blobs(n_samples = 1000,
                         n_features = 2,
                         random_state = 1)
## Standardize the features
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)

print("Features: \n", standardized_features)

## Replace the first feature's first value with a missing value
true_value = standardized_features[0,0]
standardized_features[0,0] = np.nan
## Predict the missing values in the feature matrix
#features_knn_imputed = KNeighborsClassifier( n_neighbors=5).predict(standardized_features)
## Compare true and imputed values
#print("True Value:", true_value)
#print("Imputed Value:", features_knn_imputed[0,0])

# Load library
from sklearn.impute import SimpleImputer
# Create imputer
mean_imputer = SimpleImputer(strategy="mean", verbose=0)
# Impute values

features_mean_imputed = mean_imputer.fit_transform(features)
# Compare true and imputed values
print("True Value:", true_value)
print("Imputed Value:", features_mean_imputed[0,0])





























