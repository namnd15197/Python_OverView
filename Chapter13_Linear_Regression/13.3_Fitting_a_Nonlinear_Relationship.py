
#load library
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures

#load data with one feature
boston = load_boston()

features = boston.data[:,0:1]
target = boston.target


#create polynomial features x^2 and x^3

polynomial = PolynomialFeatures(degree=3, include_bias=False)
features_polynomial = polynomial.fit_transform(features)

#create linear regression
regression = LinearRegression()
#fit the linear regression
model = regression.fit(features_polynomial, target)
