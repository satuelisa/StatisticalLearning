import numpy as np
from math import sqrt
from numpy.linalg import qr, inv
from scipy.stats import norm
from numpy.random import normal, uniform
from scipy.stats.distributions import chi2

np.seterr(all='raise') # see the origin of all troubles
np.set_printoptions(precision = 2, suppress = True)

high = 0.95 # correlation threshold
alpha = 0.01 # significance level for the p value
z1a = norm.ppf(1 - alpha) # gaussian percentile 

n = 10
p = 5
# also multiple outputs now
k = 3 
print(f'{n} inputs, {p} features')
coeff = uniform(size = (k, p), low = -5, high = 5) # coefficients for the k output
constants = np.ones((1, n))

def gen(x): # a bit more randomness this time and use integers
    count = np.shape(x)[0]
    noise = normal(loc = 0, scale = 0.1, size = count).tolist()
    return [sum([coeff[j, i] * x[i] + noise[i] for i in range(count)]) for j in range(k)]

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
    y = np.asarray(np.apply_along_axis(gen, axis = 1, arr = X)) # generate the labels using the features    
    Q, R = qr(X)
    Ri = inv(R)
    Qt = Q.T
    RiQt = np.matmul(Ri, Qt)
    w = np.matmul(RiQt, y)
    for i in range(k):    
        for j in range(p):
            print(f'Coefficient {coeff[i, j]:.2f} of output {i + 1} was estimated as {w[j, i]:.2f}')
    QQt = np.matmul(Q, Qt)
    yp = np.matmul(QQt, y)
    for i in range(n):
        print(f'Expected {y[i, :]}, predicted {yp[i, :]}')
    break # done


