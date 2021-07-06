import numpy as np
from numpy.linalg import qr, inv
from numpy.random import normal, uniform

np.seterr(all='raise') 
np.set_printoptions(precision = 2, suppress = True)

high = 0.95 
alpha = 0.01 

c = 5 # cross-validate five times
n = c * 20 # use 80 samples to train and 20 to validate on each iteration
p = 5
coeff = uniform(size = p, low = -15, high = 15) 
constants = np.ones((1, n))

def predict(X, y):
    Q, R = qr(X)
    Ri = inv(R)
    Qt = Q.T
    RiQt = np.matmul(Ri, Qt)
    w = np.matmul(RiQt, y)
    QQt = np.matmul(Q, Qt)
    return np.matmul(QQt, y)

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

e = list() # errors go here
for r in range(c):
    # since they are randomly generated, take every cth column starting at r        
    Xc = X[r::c] 
    yc = y[r::c]
    yp = predict(Xc, yc)
    diff = (yc - yp)
    e.append(np.inner(diff, diff)) # SSQ
print('Iterations', np.array(e))
print('Estimated prediction error', sum(e) / c) # average over them
    


