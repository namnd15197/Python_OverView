
#load libraries
from sklearn.linear_model import LogisticRegressionCV
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

#load data
iris = datasets.load_iris()
features = irs.data
target = iris.target

#Standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

#create decision tree classifier object
logistic_regression = LogisticRegressionCV(penalty='l2', Cs=10, random_state=0, n_jobs=-1)

#train model
model = logistic_regression.fit(features_standardized)