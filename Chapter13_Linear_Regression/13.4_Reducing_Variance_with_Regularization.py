
#load libraries
from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

#load data
boston = load_boston()
features = boston.data
target = boston.target

#standardize features
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

#create ridge regression with an alpha value
regression = Ridge(alpha=0.5)

#fit the linear regression
model = regression.fit(features_standardized, target)

# Load library
from sklearn.linear_model import RidgeCV
# Create ridge regression with three alpha values
regr_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])
# Fit the linear regression
model_cv = regr_cv.fit(features_standardized, target)
# View coefficients
print(model_cv.coef_)

#view alpha
print(model_cv.alpha_)