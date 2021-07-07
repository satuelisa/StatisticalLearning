import numpy as np
from math import fabs
import matplotlib.pyplot as plt
from random import random, randint
from numpy.random import normal, uniform

low = -7
high = 12
xt = np.arange(low, high, 0.1) 
n = len(xt)
m = 0.1
x = np.sort(xt + normal(size = n, scale = m)) 
y = np.cos(xt) - 0.2 * xt + normal(size = n, scale = m) # add noise

# some random positions to estimate
xq = np.sort(uniform(low, high, size = n // 2)) # sorted
yq = [] # we put their estimates here

k = 10 # using this many neighbors
for point in xq: # local models
    nd = [float('inf')] * k
    ny = [None] * k 
    for known in range(n):
        d = fabs(point - x[known]) # how far is the observation
        i = np.argmax(nd)
        if d < nd[i]: # if smaller than the largest of the stored
            nd[i] = d # replace that distance
            ny[i] = y[known]
    yq.append(sum(ny) / k)

plt.plot(xq, yq, c = 'orange', linewidth = 2, linestyle = 'dashed') # model
plt.scatter(xq, yq, c = 'red', s = 15) # query points
plt.scatter(x, y, c = 'gray', s = 15) # data
plt.savefig('knn2.png')

