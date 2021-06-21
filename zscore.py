import numpy as np
from math import sqrt
from numpy.linalg import inv
from scipy.stats import norm
from numpy.random import normal, uniform
from scipy.stats.distributions import chi2

np.seterr(all='raise') # see the origin of all troubles
np.set_printoptions(precision = 2, suppress = True)

high = 0.95 # correlation threshold
alpha = 0.01 # significance level for the p value
z1a = norm.ppf(1 - alpha) # gaussian percentile 

# more inputs this time so that n - 1 > p 
n = 100
p = 10
print(f'{n} inputs, {p} features')
dof = n - p - 1 # degrees of freedom
coeff = uniform(size = p, low = -5, high = 5) # coefficients of features (model)
coeff[p // 2] = 0 # make the middle one irrelevant
constants = np.ones((1, n))

def rss(X, y, w):
    yp = np.matmul(X, w) # n predictions, one per input
    yyp = y - yp 
    return np.matmul(yyp.T, yyp)  

def gen(x): # a bit more randomness this time and use integers
    count = np.shape(x)[0]
    noise = normal(loc = 0, scale = 0.1, size = count).tolist()
    return sum([coeff[i] * x[i] + noise[i] for i in range(count)]) 

while True: # iterate until we get data that yields a model
    features = uniform(low = -50, high = 50, size = (p - 1, n))
    Xt = np.vstack((constants, features)) # put the constants on the top row
    X = Xt.T # put the inputs in columns
    assert p == np.shape(X)[1]
    assert n == np.shape(X)[0]
    
    # check for correlations in the inputs (columns of X, ignoring the constant)
    cc = np.corrcoef(X[np.ix_([1, p - 1], [1, p - 1])], rowvar = False)
    mask = np.ones(cc.shape, dtype = bool)
    np.fill_diagonal(mask, 0)
    if cc[mask].max() > high:
        print('High correlations present in the inputs, regenerating')
        continue

    # check for correlations in the features (rows of X, ignoring the constant)
    cc = np.corrcoef(X[np.ix_([1, p - 1], [1, p - 1])], rowvar = True)
    mask = np.ones(cc.shape, dtype = bool)
    np.fill_diagonal(mask, 0)
    if cc[mask].max() > high:
        print('High correlations present in the features, regenerating')
        continue

    y = np.asarray(np.apply_along_axis(gen, axis = 1, arr = X)) # generate the labels using the features    
    XtX = np.matmul(Xt, X)
    try:
        XtXi = inv(XtX)
    except:
        print('Encountered a singular matrix regardless of the correlations checks')
        continue
    v = np.diag(XtXi)
    XXX = np.matmul(XtXi, Xt)
    w = np.matmul(XXX, y)
    rss1 = rss(X, y, w)
    rd = rss1 / dof
    yp = np.matmul(X, w)
    dy = y - yp
    dsq = np.inner(dy, dy) # sum([i**2 for i in dy])
    var = dsq / dof # variance
    sd = sqrt(var) # standard deviation
    assert min(v) > 0 # all the vj are positive (that is, the math worked as intended)
    for j in range(p):
        sqv = sqrt(v[j])
        ss = sd * sqv            
        z = w[j] / ss
        excl = np.copy(w)
        excl[j] = 0 # constrain this one to zero
        rss0 = rss(X, y, excl) # a model without this feature
        f = (rss0 - rss1) / rd # just one parameter was excluded
        print(z, f) # these should be equal but I am clearly doing something wrong (lemme know if you know what that is)
        p = chi2.sf(f, dof) # survival function (assuming f == z) 
        signif = p < alpha
        print(f'\nCoefficient {coeff[j]:.2f} was estimated as {w[j]:.2f}',
              'which is significant' if signif else 'but it is insignificant',
              f'with a p-value of {p:.5f}')
        if signif:
            width = z1a * sd
            low = w[j] - width
            high = w[j] + width
            print(f'with a confidence interval [{low}, {high}]',
                  'that contains zero' if low < 0 and high > 0 else '')
    break # done


