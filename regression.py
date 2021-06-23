import numpy as np    
from numpy.linalg import inv # matrix inverse
from numpy.random import uniform, normal

def rss(X, y, w):
    yp = np.matmul(X, w) # n predictions, one per input
    yyp = y - yp 
    return np.matmul(yyp.T, yyp)  

Xt = np.array([[1, 1, 1, 1, 1, 1, 1, 1], # the constants
               [2, 5, 7, 3, 5, 2, 1, 2], # feature 1
               [8, 6, 3, 1, 9, 4, 3, 2], # feature 2
               [2, 3, 5, 2, 4, 5, 7, 3], # feature 3
               [3, 4, 5, 4, 4, 8, 8, 2]]) # feature 4
X = Xt.T
n = np.shape(X)[0]
p = np.shape(X)[1]
print(p, 'features (incl. a constant)')
print(n, 'inputs')
coeff = uniform(size = p, low = -5, high = 5) # a model

def gen(x): # a bit more randomness this time and use integers
    count = np.shape(x)[0]
    noise = normal(loc = 0, scale = 0.1, size = count).tolist()
    return sum([coeff[i] * x[i] + noise[i] for i in range(count)]) 

y = np.asarray(np.apply_along_axis(gen, axis = 1, arr = X)) 
print(f'{n} labels: {y}')    
XtX = np.matmul(Xt, X)
XtXi = inv(XtX)
XXX = np.matmul(XtXi, Xt)
w = np.matmul(XXX, y)
ba = rss(X, y, w)
print(f'best analytical {ba:.3f} with weights {w}:')
for (wp, wt) in zip(w, coeff):
    print(f'{wp:.3f} (recovered) for {wt:.3f} (inserted)')

pred = np.matmul(X, w)
print('predictions:')
for (yl, yp) in zip(y, pred):
    print(f'{yp:.3f} for {yl:.3f}')
    
lowest = float('inf')
for r in range(1000): # try a bunch of random ones
    wr = np.array(uniform(low = min(w), high = max(w), size = p))
    lowest = min(lowest, rss(X, y, wr))
print(f'the best random attempt has RSS {lowest:.2f} whereas the optimal has {ba:.2f}')
