import numpy as np    
from numpy.linalg import inv 
from numpy.random import normal

np.set_printoptions(precision = 0, suppress = True)
Xt = np.array([[1, 1, 1, 1, 1, 1, 1, 1], # constant
               [2, 5, 7, 3, 5, 2, 1, 2], # feature 1
               [8, 6, 3, 1, 9, 4, 3, 2], # feature 2
               [2, 3, 5, 2, 4, 5, 7, 3], # feature 3
               [3, 4, 5, 4, 4, 8, 8, 2]]) # feature 4
X = Xt.T
coeff = np.array([0, 1, 2, 3, 4]) # zero weight for the constant
threshold = 30 # make two classes

def gen(x): # a bit more randomness this time and use integers
    count = np.shape(x)[0]
    noise = normal(loc = 0, scale = 0.1, size = count).tolist()
    value = sum([coeff[i] * x[i] + noise[i] for i in range(count)])
    return (value > threshold) + 1 # class one or class two

y = np.asarray(np.apply_along_axis(gen, axis = 1, arr = X))
Y = np.array([(1.0 * (y == 1)), (1.0 * (y == 2))]).T
XtXi = inv(np.matmul(Xt, X))
XXtXi = np.matmul(X, XtXi)
XXtXiXt = np.matmul(XXtXi, Xt)
Yp = np.matmul(XXtXiXt, Y)
print('Assigned indicators')
print(Y.T)
print('Predicted indicators')
print(np.fabs(Yp.T)) # the -0 bother me so I use fabs

