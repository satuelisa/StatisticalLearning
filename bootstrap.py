import numpy as np
from random import choices # sample with replacement
from numpy.random import normal, uniform
from sklearn.linear_model import LinearRegression

np.seterr(all='raise') 
np.set_printoptions(precision = 2, suppress = True)
high = 0.95 

b = 5 # bootstrap five times
n = 100 # again with 100 input total
p = 5
coeff = uniform(size = p, low = -15, high = 15) 
constants = np.ones((1, n))

def gen(x): 
    count = np.shape(x)[0]
    noise = normal(loc = 0, scale = 0.1, size = count).tolist()
    return sum([coeff[i] * x[i] + noise[i] for i in range(count)])

features = None
while True: # iterate until we get data that yields a model
    features = uniform(low = -50, high = 50, size = (p - 1, n))
    Xt = np.vstack((constants, features)) # put the constants on the top row
    X = Xt.T # put the inputs in columns
    cc = np.corrcoef(X[np.ix_([1, p - 1], [1, p - 1])], rowvar = False)
    mask = np.ones(cc.shape, dtype = bool)
    np.fill_diagonal(mask, 0)
    if cc[mask].max() > high:
        print('High correlations present in the inputs, regenerating')
        continue
    cc = np.corrcoef(X[np.ix_([1, p - 1], [1, p - 1])], rowvar = True)
    mask = np.ones(cc.shape, dtype = bool)
    np.fill_diagonal(mask, 0)
    if cc[mask].max() > high:
        print('High correlations present in the features, regenerating')
        continue
    y = np.asarray(np.apply_along_axis(gen, axis = 1, arr = X))
    break

baseline = LinearRegression().fit(X, y) 
print(baseline.coef_) # inspect the original model

pos = [i for i in range(n)]
for r in range(b): # independent replicas
    Xb = np.zeros((n, p))
    yb = np.zeros(n)
    i = 0
    for s in choices(pos, k = n): # sample with replacement
        Xb[i, :] = X[s, :]
        yb[i] = y[s]
    model = LinearRegression().fit(Xb, yb) # our simple method would cry about singular matrices
    print(model.coef_) # check out the resulting model
    


