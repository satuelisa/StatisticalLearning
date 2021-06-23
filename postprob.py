import numpy as np    
from math import exp
from numpy.linalg import qr, inv 
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

intended = np.asarray(np.apply_along_axis(gen, axis = 1, arr = X))
print('Intended labels')
print(intended)
y = np.array([(1.0 * (intended == 1)), (1.0 * (intended == 2))]).T
Q, R = qr(X)
Ri = inv(R)
Qt = Q.T
RiQt = np.matmul(Ri, Qt)
w = np.matmul(RiQt, y)
QQt = np.matmul(Q, Qt)
yp = np.matmul(QQt, y)
print('Assigned indicators')
print(y.T)
print('Predicted indicators (rounded)')
print(np.fabs(np.round(yp, decimals = 0)).T)
wt = w.T
for c in [1, 2]: # for both classes
    b = wt[c - 1]
    print('Class', c)
    for i in range(len(intended)):
        x = X[i]
        yd = intended[i]
        eb = exp(np.matmul(b, x)) 
        ed = 1 + eb
        pThis = eb / ed
        pOther = 1 / ed
        ok = pThis > pOther
        print(f'{x} should be {yd}: this {pThis:.2f} vs. other {pOther:.2f} ->', 'right' if ok else 'wrong')
    

