
#load libraries
from scipy.stats import uniform
from sklearn import linear_model, datasets
from sklearn.model_selection import RandomizedSearchCV

#load data

iris = datasets.load_iris()

features = iris.data
target = iris.target

logistic = linear_model.LogisticRegression()

#create range of candidate regularization penalty hyperparameter values
penalty = ['l1', 'l2']

#create distribution of candidate regularization hyperparameter values
C = uniform(loc=0, scale=4)

#create hyperparameter options
hyperparameter = dict(C=C, penalty=penalty)
#create randomized search

randomizedsearch = RandomizedSearchCV(
    logistic, hyperparameter, random_state=1, n_iter=100, cv=5, verbose=0)

#fit randomized search

best_model = randomizedsearch.fit(features, target)

print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

