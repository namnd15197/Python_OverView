#20.1 Preprocessing Data for Neural Networks

#load libraries
from sklearn import preprocessing
import numpy as np

#create features
features = np.array([[-100.1, 3240.1],
                     [-200.2, -243.1],
                     [5000.5, 150.1],
                     [6000.6, -125.1],
                     [9000.9, -673.1]])

#create scaler
scaler = preprocessing.StandardScaler()

#transform the features
features_standardized = scaler.fit_transform(features)

#show features
print(features_standardized)

print("Mean: ", round(features_standardized[:,0].mean()))
print("Standard deviation: ", features_standardized[:,0].std())

