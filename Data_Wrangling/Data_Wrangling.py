
#3.0 Introduction
# Load library
import pandas as pd
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'
# Load data as a dataframe
#dataframe = pd.read_csv(url)
#print(dataframe.shape)
# Show first 5 rows
#print(dataframe.head(5))


#3.1 Creating a Data Frame

# Load library
import pandas as pd
# Create DataFrame
dataframe = pd.DataFrame()
# Add columns
dataframe['Name'] = ['Jacky Jackson', 'Steven Stevenson']
dataframe['Age'] = [38, 25]
dataframe['Driver'] = [True, False]

# Create row
new_person = pd.Series(['Molly Mooney', 40, True], index=['Name','Age','Driver'])
# Append row
dataframe.append(new_person, ignore_index=True)

# Show DataFrame
#print(dataframe)



#3.2 Describing the Data
# Load library
import pandas as pd
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'
# Load data as a dataframe
dataframe = pd.read_csv(url)
#print(dataframe)

# Show first 2 rows
#print(dataframe.head(2))

# Show dimensions
#print(dataframe.shape)

# Show statistics
#print(dataframe.describe())


#3.3 Navigating DataFrames
# Load library
import pandas as pd
# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'
# Load data as a dataframe
dataframe = pd.read_csv(url)
# Select first row
#print(dataframe.iloc[0])

# Select three rows
#print(dataframe.iloc[:4])

# Set index
dataframe = dataframe.set_index(dataframe['Name'])

# Show row
#print(dataframe.loc['Allison, Mr Hudson Joshua Creighton'])

#loc is useful when the index of the DataFrame is a label (e.g., a string).
#iloc works by looking for the position in the DataFrame. For example, iloc[0]
#will return the first row regardless of whether the index is an integer or a label.

#3.4 Selecting Rows Based on Conditionals


# Create URL
url = 'https://raw.githubusercontent.com/chrisalbon/simulated_datasets/master/titanic.csv'
# Load data as a dataframe
dataframe = pd.read_csv(url)
# Show top two rows where column 'sex' is 'female'
#print(dataframe[dataframe['Sex'] == 'female'].head(2))

# Filter rows
#print(dataframe[(dataframe['Sex'] == 'female') & (dataframe['Age'] >= 65)])

#3.5 Replacing Values

# Load data as a dataframe
dataframe = pd.read_csv(url)
# Replace values, show two rows
#print(dataframe['Sex'].replace("female", "Woman").head(2))

# Replace "female" and "male with "Woman" and "Man"
#print(dataframe['Sex'].replace(["female", "male"], ["Woman", "Man"]).head(5))

# Replace values, show ten rows
#print(dataframe.replace(1, "One").head(10))

# Replace values, show two rows
#print(dataframe.replace(r"1st", "First", regex=True).head(2))



#3.6 Renaming Columns

# Rename column, show two rows
#print(dataframe.rename(columns={'PClass': 'Passenger Class'}).head(2))

# Load library
import collections
# Create dictionary
column_names = collections.defaultdict(str)
# Create keys
for name in dataframe.columns:
 column_names[name]
# Show dictionary
#print(column_names)

#3.7 Finding the Minimum, Maximum, Sum, Average, and Count

# Calculate statistics
#print('Maximum:', dataframe['Age'].max())
#print('Minimum:', dataframe['Age'].min())
#print('Mean:', dataframe['Age'].mean())
#print('Sum:', dataframe['Age'].sum())
#print('Count:', dataframe['Age'].count())

# Show counts
#print(dataframe.count())







#3.8 Finding Unique Values

# Select unique values
#print(dataframe['Sex'].unique())

# Show counts
#print(dataframe['Sex'].value_counts())

# Show counts
#print(dataframe['PClass'].value_counts())



#3.9 Handling Missing Values

## Select missing values, show two rows
#print('age with nan \n', dataframe[dataframe['Age'].isnull()].head(2))

# Attempt to replace values with NaN
import numpy as np
dataframe['Age'] = dataframe['Age'].replace(np.nan, 0)
#print("age without nan \n", dataframe[dataframe['Age'].isnull()].head(2))
#dataframe = pd.read_csv(url, na_values=[np.nan, 'NONE', -999])

# Load data, set missing values
#dataframe = pd.read_csv(url, na_values=[np.nan, 'NONE', -999])


#3.10 Deleting a Column

# Load data
dataframe = pd.read_csv(url)
# Delete column
#print('Delete column \n', dataframe.drop('Age', axis=1).head(2))

# Drop columns
#print('Delete columns \n', dataframe.drop(['Age', 'Sex'], axis=1).head(2))

# Drop column without name
#print('Delete column without name \n', dataframe.drop(dataframe.columns[1], axis=1).head(2))


#3.11 Deleting a Row

# Load data
#dataframe = pd.read_csv(url)
# Delete rows, show first two rows of output
dataframe[dataframe['Sex'] != 'male'].head(2)

# Delete row, show first two rows of output
dataframe[dataframe['Name'] != 'Allison, Miss Helen Loraine'].head(2)

# Delete row, show first two rows of output
dataframe[dataframe.index != 0].head(2)



#3.12 Dropping Duplicate Rows

# Load data
dataframe = pd.read_csv(url)
# Drop duplicates, show first two rows of output
print(dataframe.drop_duplicates().head(2))

# Show number of rows
print("Number Of Rows In The Original DataFrame:", len(dataframe))
print("Number Of Rows After Deduping:", len(dataframe.drop_duplicates()))



#3.13 Grouping Rows by Values

# Load data
dataframe = pd.read_csv(url)
# Group rows by the values of the column 'Sex', calculate mean
# of each group
dataframe.groupby('Sex').mean()

# Group rows, count rows
dataframe.groupby('Survived')['Name'].count()

# Group rows, calculate mean
print(dataframe.groupby(['Sex','Survived'])['Age'].mean())

#3.14 Grouping Rows by Time

# Create date range
time_index = pd.date_range('06/06/2017', periods=100000, freq='30S')
# Create DataFrame
dataframe = pd.DataFrame(index=time_index)
# Create column of random values
dataframe['Sale_Amount'] = np.random.randint(1, 10, 100000)
# Group rows by week, calculate sum per week
#print(dataframe.resample('W').sum())
# Show three rows
dataframe.head(3)
# Group by two weeks, calculate mean
dataframe.resample('2W').mean()
# Group by month, count rows
dataframe.resample('M').count()
# Group by month, count rows
dataframe.resample('M', label='left').count()


#3.15 Looping Over a Column
# Load data
dataframe = pd.read_csv(url)
# Print first two names uppercased
for name in dataframe['Name'][0:2]:
 print(name.upper())

 # Show first two names uppercased
print([name.upper() for name in dataframe['Name'][0:2]])





#3.16 Applying a Function Over All Elements in a Column

# Load data
dataframe = pd.read_csv(url)

# Create function
def uppercase(x):
 return x.upper()

# Apply function, show two rows
print(dataframe['Name'].apply(uppercase)[0:2])


#3.17 Applying a Function to Groups
# Load data
dataframe = pd.read_csv(url)
# Group rows, apply function to groups
print(dataframe.groupby('Sex').apply(lambda x: x.count()))

#3.18 Concatenating DataFrames

data_a = {'id': ['1', '2', '3'],
 'first': ['Alex', 'Amy', 'Allen'],
 'last': ['Anderson', 'Ackerman', 'Ali']}
dataframe_a = pd.DataFrame(data_a, columns = ['id', 'first', 'last'])
# Create DataFrame
data_b = {'id': ['4', '5', '6'],
 'first': ['Billy', 'Brian', 'Bran'],
 'last': ['Bonder', 'Black', 'Balwner']}
dataframe_b = pd.DataFrame(data_b, columns = ['id', 'first', 'last'])
# Concatenate DataFrames by rows
# axis 0 la content theo hang, axis 1 la content theo cot
print(pd.concat([dataframe_a, dataframe_b], axis=1))

# Create row
row = pd.Series([10, 'Chris', 'Chillon'], index=['id', 'first', 'last'])
# Append row
print(dataframe_a.append(row, ignore_index=True))


#3.19 Merging DataFrames

# Create DataFrame
employee_data = {'employee_id': ['1', '2', '3', '4'],
                 'name': ['Amy Jones', 'Allen Keys', 'Alice Bees',
                 'Tim Horton']}
dataframe_employees = pd.DataFrame(employee_data, columns = ['employee_id','name'])
# Create DataFrame
sales_data = {'employee_id': ['3', '4', '5', '6'],
              'total_sales': [23456, 2512, 2345, 1455]}
dataframe_sales = pd.DataFrame(sales_data, columns = ['employee_id', 'total_sales'])
# Merge DataFrames
pd.merge(dataframe_employees, dataframe_sales, on='employee_id')

# Merge DataFrames
pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='outer')

# Merge DataFrames
print(pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='left'))

#Inner
#Return only the rows that match in both DataFrames (e.g., return any row with
#an employee_id value appearing in both dataframe_employees and data
#frame_sales).
#Outer
#Return all rows in both DataFrames. If a row exists in one DataFrame but not in
#the other DataFrame, fill NaN values for the missing values (e.g., return all rows
#in both employee_id and dataframe_sales).
#Left
#Return all rows from the left DataFrame but only rows from the right DataFrame
#that matched with the left DataFrame. Fill NaN values for the missing values (e.g.,
#return all rows from dataframe_employees but only rows from data
#frame_sales that have a value for employee_id that appears in data
#frame_employees).
#Right
#Return all rows from the right DataFrame but only rows from the left DataFrame
#that matched with the right DataFrame. Fill NaN values for the missing values
#(e.g., return all rows from dataframe_sales but only rows from data
#frame_employees that have a value for employee_id that appears in data
#frame_sales).

















