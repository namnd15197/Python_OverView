#load libraries

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

#load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

#standardize features
scaler = StandardScaler()

features_standardized = scaler.fit_transform(features)

#create LogisticRegression
logistic_regression = LogisticRegression(random_state = 0, multi_class="ovr")

#train model
model = logistic_regression.fit(features_standardized, target)


