#load libraries

from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets

#load data
boston = datasets.load_boston()
features = boston.data[:,0:2]
target = boston.target

#create RandomForestRegressor
regressor = RandomForestRegressor(random_state = 0, n_jobs=-1)

model_mse = regressor.fit(features, target)

observation = observation = [[0.02, 16]]
print("MSE: ", model_mse.predict(observation))

regressor_mae = RandomForestRegressor(criterion='mae', random_state=0, n_jobs=-1)
model_mae = regressor_mae.fit(features, target)
print("MAE: ", model_mae.predict(observation))

