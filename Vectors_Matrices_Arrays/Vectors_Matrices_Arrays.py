
#Use numpy to create a one - dimensional Array

#load library
import numpy as np

#Create a vector as a row
vector_row = np.array([1, 2, 3])

#create a vector as a column
vector_column = np.array([[1], [2], [3]])


#Use numpy to create a two - dimensional array:

#create a matrix
matrix = np.array([[1, 2], [1, 2], [1, 2]])


#Creating a Sparse Matrix

from scipy import sparse

#Create a matrix
matrix_temp = np.array([[0,0], [0,1], [3, 0]])

#print(matrix_temp)

#Create compressed sparse row (CSR) matrix
matrix_sparse = sparse.csr_matrix(matrix_temp)

#view sparse matrix
#print(matrix_sparse)

#create larger matrix
matrix_large = np.array([[0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0],
                         [3, 0, 0, 0, 0, 0, 0]])

#create compressed sparse row (CSR) matrix
matrix_large_sparse = sparse.csr_matrix(matrix_large)

#view original sparse matrix
#print(matrix_large)

#view larger sparse matrix
#print(matrix_large_sparse)



#Select one or more elements in a vector or matrix

#create row vector
vector = np.array([1, 2, 3, 4, 5, 6])

#Create matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

#select third element of vector
print(vector[2])

#select second row, second column
print(matrix[1,1])

#Select all elements of a vector
print(vector[:])

#select everythig up to and including the third element
print(vector[:3])

#select everything after the third elemennt
print(vector[3:])

#Select the last element
print(vector[-1])

#select the first two rows and all columns of a matrix
print(matrix[:2,:])

#select all rows and the second column
print(matrix[:,1:2])


#
print(matrix[1:2,1:3])
# a: gia tri nho nhat cua hang = 1
# b: gia tri lon nhat cua hang = matrix.rows
# c: gia tri nho nhat cua cot = 1
# d: gia tri lon nhat cua cot = matrix.cols


#Describe the shape, size, and dimensions of the matrix
#Create matrix
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])


#View number of rows and columns
print(matrix.shape)

#view number of elements (rows*cols)
print(matrix.size)

#view number of dimensions
print(matrix.ndim)


#create matrix
matrix = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

#Create function that adds 100 to something

add_100 = lambda i: i+ 100

#create vectorized function
vectorized_add_100 = np.vectorize(add_100)

#apply function to all elements in matrix
print(vectorized_add_100(matrix))

print(matrix + 100)

#return maximum element
print(np.max(matrix))

#return minimum element
print(np.min(matrix))

#find maximum element in each column
print(np.max(matrix, axis=0))

#find maximum element in each row
print(np.max(matrix, axis=1))


#1.8 Calculating the Average, Variance, and Standard Deviation

#Return mean
print(matrix.mean)

#return variance(Phương sai)
print(np.var(matrix))

#return standard deviation(Độ lệch chuẩn)
print(np.std(matrix))

#find the mean value in each row
print(np.mean(matrix, axis = 0))


#1.9 Reshaping Arrays
#create 4x3 matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])

#reshape matrix into 2x6 matrix
print(matrix.reshape(2, 6))

#reshape to one row, argument in reshape is -1, which efectively means "as many as needed."
print(matrix.reshape(2, -1))

#one argument
print(matrix.reshape(12))

#1.10 Transposing a Vector or Matrix

#transpose matrix
print(matrix.T)

#transpose vector ( cannot transpose vector because it is just a collection of value)
print(vector.T)

#transpose row vector
print(np.array([[1, 2, 3, 4, 5, 6]]).T)


#1.11 Flattening a Matrix, we can use reshape in another way

print(matrix.flatten())


#1.12 Find the Rank of a Matrix

print(np.linalg.matrix_rank(matrix))


#1.13 Calculating the Determinant(định thức)

matrix = np.array([[1, 2, 3],
                    [2, 4, 6],
                 [3, 8, 9]])
#return determinant of matrix
print(np.linalg.det(matrix))



#1.14 Getting the Diagonal of a Matrix ( đường chéo )

#return diagonal elements
print(matrix.diagonal())

# Return diagonal one above the main diagonal
print(matrix.diagonal(offset=1))


# Return diagonal one below the main diagonal
print(matrix.diagonal(offset=-1))



#1.15 Calculating the Trace of a Matrix
# tong gia tri tren duong cheo chinh

print(matrix.trace())
print(sum(matrix.diagonal()))


#1.16 Finding Eigenvalues and Eigenvectors

#create Matrix
matrix = np.array([[1, -1, 3],
                   [1, 1, 6],
                   [3, 8, 9]])

#calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)

#view eigenvalues, eigenvectors
print(eigenvalues)
print(eigenvectors)


#1.17 Calculating Dot Products

# Create two vectors
vector_a = np.array([1,2,3])
vector_b = np.array([4,5,6])

# Calculate dot product
print(np.dot(vector_a, vector_b))

# Calculate dot product Upper Py 3.5
print(vector_a @ vector_b) 



#1.18 Adding and Subtracting Matrices
# Create matrix
matrix_a = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 2]])
# Create matrix
matrix_b = np.array([[1, 3, 1],
                     [1, 3, 1],
                     [1, 3, 8]])
# Add two matrices
print(np.add(matrix_a, matrix_b))
print(matrix_a + matrix_b)
# Subtract two matrices
print(np.subtract(matrix_a, matrix_b))
print(matrix_a - matrix_b)


#1.19 Multiplying Matrices
# Create matrix
matrix_a = np.array([[1, 1],
                     [1, 2]])
# Create matrix
matrix_b = np.array([[1, 3],
                     [1, 2]])
# Multiply two matrices
print(np.dot(matrix_a, matrix_b))
print(matrix_a * matrix_b)


#1.20 Inverting a Matrix
# Create matrix
matrix = np.array([[1, 4],
                   [2, 5]])
# Calculate inverse of matrix
print(np.linalg.inv(matrix))
# Multiply matrix and its inverse
print(matrix @ np.linalg.inv(matrix))


#1.21 Generating Random Values

# Set seed
np.random.seed(0)
# Generate three random floats between 0.0 and 1.0
print(np.random.random(3))

# Generate three random integers between 1 and 10
print(np.random.randint(0, 11, 3))

# Draw three numbers from a normal distribution with mean 0.0
# and standard deviation of 1.0
print(np.random.normal(0.0, 1.0, 3))

# Draw three numbers from a logistic distribution with mean 0.0 and scale of 1.0
print(np.random.logistic(0.0, 1.0, 3))

# Draw three numbers greater than or equal to 1.0 and less than 2.0
print(np.random.uniform(1.0, 2.0, 3))


