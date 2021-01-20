
# Load libraries
from sklearn.datasets import load_boston
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split

# Load data
boston = load_boston()

# Create features
features, target = boston.data, boston.target

# Make test and training split
features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=0)


# Create a dummy regressor
dummy = DummyRegressor(strategy='mean')
# "Train" dummy regressor
dummy.fit(features_train, target_train)
# Get R-squared score
print(dummy.score(features_test, target_test))


#load library
from sklearn.linear_model import LinearRegression

#train simple linear regression model
ols = LinearRegression()
ols.fit(features_train, target_train)

#get R-squared score
print(ols.score(features_test, target_test))







