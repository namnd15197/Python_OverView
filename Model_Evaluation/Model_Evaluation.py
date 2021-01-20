
#Load libraries
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

#load digit datasets
digits = datasets.load_digits()
#print(digits)
#Create features matrix
features = digits.data

#create target vector
target = digits.target

#create standardizer
standardizer = StandardScaler()

#Create logistic regression object
logit = LogisticRegression()

#Create a pipeline that standardizes, then runs logistic regression
pipeline = make_pipeline(standardizer, logit)

#Create k-fold cross - validation
kf = KFold(n_splits=10, shuffle=True, random_state=1)

#conduct k - fold cross validation
cv_results = cross_val_score(pipeline, features, target, cv=kf, scoring="accuracy", n_jobs=-1)
cv_results.mean()


# View score for all 10 folds
cv_results
