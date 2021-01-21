
#load library
from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

#load data
boston = load_boston()

features = boston.data
target = boston.target

#standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

#create lasso regression with alpha value
regression = Lasso(alpha=0.5)

#fit the linear regression
model = regression.fit(features_standardized, target)

#view coefficients
print(model.coef_)

#create lasso regression with a high alpha
regression_a10 = Lasso(alpha=10)
model_a10 = regression_a10.fit(features_standardized, target)
print(model_a10.coef_)



