
#load libraries
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

#load  data with only two features
boston = load_boston()

features = boston.data[:,0:2]
target = boston.target

#create linear regression
regression = LinearRegression()
#fit the linear regression
model = regression.fit(features, target)

