import numpy as np    
from numpy.random import uniform, normal

x = np.array([1, 2, 3, 4]).T # a column vector n x 1
n = len(x) # (n - 1) features plus the constant gives n
w = np.array(uniform(size = n)).T # weights (random for now, between 0 and 1 for starters)
iyp = np.inner(x, w) # inner product of two arrays

xt = [ x.T ] # transpose into a row vector, 1 x n
yp = np.matmul(xt, w) # a scalar prediction, (1 x n) x (n x 1) = 1 x 1
assert iyp == yp # should coincide with the inner product from above

X = np.array([[1, 1, 1], # constants
              [3, 5, 7], # the first feature
              [8, 7, 3],
              [2, 4, 6]]) # n x p in total
assert n == np.shape(X)[0] # features as rows
p = np.shape(X)[1] # inputs as columns

# let assume the model is 3 x1 - 2 x2 + 4 x3 - 5 with small gaussian noise
def gen(x): # generate integer labels from an arbitrary model
    label = 5 * x[0] + 3 * x[1] - 2 * x[2] + 4 * x[3] \
                  + normal(loc = 0, scale = 0.2, size = 1)
    label = round(label[0])
    print(x, 'gets', label)
    return label
    
y = np.apply_along_axis(gen, axis = 0, arr = X) # 1 x p, one per input (rows)

def rss(X, y, w):
    Xt = X.T
    yp = np.matmul(Xt, w) # predictions for all inputs
    yyp = y - yp
    assert np.shape(yp) == np.shape(y)
    return np.matmul(yyp.T, yyp)

for r in range(10): # replicas
    wr = np.array(uniform(low = -6, high = 6, size = n)).T
    print(f'{rss(X, y, wr)} for {wr}') # the smaller the better (all will be horrible)
