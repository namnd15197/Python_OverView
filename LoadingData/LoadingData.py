

#2.1 Loading a Sample Dataset


# Load scikit-learn's datasets
from sklearn import datasets
# Load digits dataset
digits = datasets.load_digits()
# Create features matrix
features = digits.data
# Create target vector
target = digits.target
# View first observation
print(features[0])



#2.2 Creating a Simulated Dataset
# Load library
from sklearn.datasets import make_regression
# Generate features matrix, target vector, and the true coefficients
features, target, coefficients = make_regression(n_samples = 100,
                                                 n_features = 3,
                                                 n_informative = 3,
                                                 n_targets = 1,
                                                 noise = 0.0,
                                                 coef = True,
                                                 random_state = 1)


# View feature matrix and target vector
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])


# Load library
from sklearn.datasets import make_classification
# Generate features matrix and target vector
features, target = make_classification(n_samples = 100,
                                         n_features = 3,
                                         n_informative = 3,
                                         n_redundant = 0,
                                         n_classes = 2,
                                         weights = [.25, .75],
                                         random_state = 1)
# View feature matrix and target vector
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])


# Load library
from sklearn.datasets import make_blobs
# Generate feature matrix and target vector
features, target = make_blobs(n_samples = 100,
                             n_features = 2,
                             centers = 3,
                             cluster_std = 0.5,
                             shuffle = True,
                             random_state = 1)
# View feature matrix and target vector
print('Feature Matrix\n', features[:3])
print('Target Vector\n', target[:3])


#2.3 Loading a CSV File

# Load library
import pandas as pd
# Create URL
url = 'https://tinyurl.com/simulated_data'
# Load dataset
#dataframe = pd.read_csv(url)
# View first two rows
#print(dataframe.head(2))


#2.4 Loading an Excel File

# Load library
import pandas as pd
# Create URL
url = 'https://tinyurl.com/simulated_excel'
# Load data
dataframe = pd.read_excel(url, sheetname=0, header=1)
# View the first two rows
dataframe.head(2)


#2.5 Loading a JSON File

# Load library
import pandas as pd
# Create URL
url = 'https://tinyurl.com/simulated_json'
# Load data
dataframe = pd.read_json(url, orient='columns')
# View the first two rows
dataframe.head(2)


#2.6 Querying a SQL Database

# Load libraries
import pandas as pd
from sqlalchemy import create_engine
# Create a connection to the database
database_connection = create_engine('sqlite:///sample.db')
# Load data
dataframe = pd.read_sql_query('SELECT * FROM data', database_connection)
# View first two rows
dataframe.head(2)