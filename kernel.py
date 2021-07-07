import numpy as np
from math import fabs
import matplotlib.pyplot as plt
from random import random, randint
from numpy.random import normal, uniform

def dt(t): # Eq. (6.4)
    return (3/4) * (1 - t**2) if fabs(t) <= 1 else 0

def kernel(qp, dp, lmd = 0.2): # Eq. (6.3)
    return dt(fabs(qp - dp) / lmd)

low = -7
high = 12
xt = np.arange(low, high, 0.1) 
n = len(xt)
m = 0.1
x = np.sort(xt + normal(size = n, scale = m)) 
y = np.cos(xt) - 0.2 * xt + normal(size = n, scale = m) 
xq = np.sort(uniform(low, high, size = n // 2)) # sorted for the visuals
yq = []
k = 10
tiny = 0.001
for point in xq:
    nd = [float('inf')] * k
    nx = [None] * k
    ny = [None] * k
    for known in range(n):
        d = fabs(point - x[known]) # how far is the observation
        i = np.argmax(nd)
        if d < nd[i]: # if smaller than the largest of the stored
            nd[i] = d # replace that distance
            nx[i] = x[known]
            ny[i] = y[known]
    w =  [kernel(point, neighbor) for neighbor in nx] # apply the kernel 
    bottom = sum(w) # the normalizer
    if fabs(bottom) > tiny: 
        top = sum((w * yv) for (w, yv) in zip(w, ny)) # weighted sum of the y values
        yq.append(top / bottom) # store the obtained value for this point
    else: # do NOT divide by zero
        yq.append(None) # no value obtained for this point (omit in drawing)

plt.rcParams.update({'text.color': 'green',
                     'xtick.color': 'green',
                     'ytick.color': 'green',
                     'axes.labelcolor': 'green',
                     'axes.edgecolor': 'green',
                     'axes.facecolor':  'none' })
plt.plot(xq, yq, c = 'orange', linewidth = 2, linestyle = 'dashed') # model
plt.scatter(xq, yq, c = 'red', s = 15) # query points
plt.scatter(x, y, c = 'gray', s = 15) # data
plt.savefig('kernel.png', transparent = True)

