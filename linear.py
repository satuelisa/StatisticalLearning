import numpy as np    
from numpy.random import uniform

x = np.transpose(np.array([1, 2, 3, 4])) # column vector 4 x 1
n = len(x) # three features plus the constant
w = uniform(size = n) # four weights (random for now)
yp = np.inner(x, w) # inner product of two rows
print(yp)

w = np.transpose(w) # also as a column vector 4 x 1
yp = np.matmul(np.transpose(x), w)  # (1 x 4) x (4 x 1) = 1 x 1
print(yp)

X = np.array([[1, 2, 3, 4], [1, 3, 5, 7], [1, 8, 7, 3]]) # 3 x 4
y = np.transpose(np.array([0.9, 1.4, 1.3])) # 3 x 1, one per input

def rss(X, y, w):
    yp = np.matmul(X, w) # predictions for all inputs
    return np.matmul(np.transpose(y - yp), (y - yp))

for r in range(10): # replicas
    print(rss(X, y, uniform(size = n))) # the smaller the better
