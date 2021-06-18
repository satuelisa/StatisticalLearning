import numpy as np
from math import sqrt
from numpy.linalg import inv
from scipy.stats import chi2, norm
from numpy.random import normal, uniform

np.set_printoptions(precision = 2, suppress = True)

def gen(x): # include two redundant inputs, x3 and x5
    count = np.shape(x)[0] + 1
    noise = normal(loc = 0, scale = 0.05, size = count).tolist()
    # a non-zero coefficient this time and no rounding
    return noise.pop() + 50 \
        + (12 + noise.pop()) * x[0] \
        + (8 + noise.pop()) * x[1] \
        - (23 + noise.pop()) * x[2] \
        + noise.pop() * x[3] \
        - (16 + noise.pop()) * x[4] \
        + noise.pop() * x[5] \
        - (4 + noise.pop()) * x[6]

n = 15
p = 7
high = 0.9 # correlation threshold
alpha = 0.05 # significance level for the p value
z1a = norm.ppf(1 - alpha) # gaussian percentile 

# more inputs this time so that n - 1 > p 
constants = np.ones((1, n))

while True: # iterate until we get data that yields a model
    features = uniform(low = -50, high = 50, size = (p - 1, n))
    X = np.vstack((constants, features)) # put the constants on the top row
    assert p == np.shape(X)[0]
    assert n == np.shape(X)[1]
    
    # check for correlations in the inputs (columns of X)
    cc = np.corrcoef(X, rowvar = False) 
    mask = np.ones(cc.shape, dtype = bool)
    np.fill_diagonal(mask, 0)
    if cc[mask].max() > high:
        print('High correlations present in the inputs, aborting')
        continue

    # check for correlations in the features (rows of X, excluding the constants)
    cc = np.corrcoef(X[np.ix_([1, p - 1], [1, p - 1])], rowvar = True)
    mask = np.ones(cc.shape, dtype = bool)
    np.fill_diagonal(mask, 0)
    if cc[mask].max() > high:
        print('High correlations present in the features, aborting')
        continue

    y = np.asarray(np.apply_along_axis(gen, axis = 1, arr = X)) # generate the labels using the features
    Xt = X.T
    XtX = np.matmul(Xt, X)
    try:
        XtXi = inv(XtX)
    except:
        print('Encountered a singular matrix regardless of the correlations checks')
        continue
    v = np.diag(XtXi)
    XXX = np.matmul(XtXi, Xt)
    w = np.matmul(XXX, y)
    yp = np.matmul(X, w)
    dof = n - p - 1
    dy = y - yp
    dsq = np.inner(dy, dy) # sum([i**2 for i in dy])
    var = dsq / dof # variance
    sd = sqrt(var) # standard deviation

    # Z-scores
    if min(v) > 0: # all the vj are positive (that is, the math worked as intended)
        for j in range(p):
            sqv = sqrt(v[j])
            ss = sd * sqv            
            z = w[j] / ss
            signif = chi2.pdf(z, dof) < alpha
            print('Index', j, 'got a coefficient', w[j], 'which is non-zero' if signif else 'which is insignificant')
            if signif:
                width = z1a * sd
                low = w[j] - width
                high = w[j] + width
                print(f'with a confidence interval [{low}, {high}]')
        break # done
    else:
        print('The regression model was not sound')    

