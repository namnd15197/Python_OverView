
#load libraries
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

#load data with only two features
boston = load_boston()
features = boston.data[:,0:2]
target = boston.target

#create linear regression

regression = LinearRegression()
#fit the linear regression
model = regression.fit(features, target)
#y = β0 + β1x1 + β2x2 + ᑙ
# View the intercept (bias β0)
print("Intercept: ", model.intercept_)

#And β1 and β2 are shown using coef_
print("Coeff: ", model.coef_)

print("Price house: ", target[0]*1000)

# Predict the target value of the first observation, multiplied by 1000
print("Price house predicted: ",model.predict(features)[0]*1000)














