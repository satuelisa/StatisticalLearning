from scipy.interpolate import CubicSpline
from numpy.random import normal
import matplotlib.pyplot as plt
import numpy as np

xt = np.arange(2, 4, 0.01) # true x coordinates
n = len(xt)
magnitude = 0.01
x = np.sort(xt + normal(size = n, scale = magnitude)) # add noise and SORT
y = np.cos(xt) + normal(size = n, scale = magnitude) 
s = CubicSpline(x, y) # as before
ns = CubicSpline(x, y, bc_type = 'natural') # fit a NATURAL cubic spline

plt.scatter(x, y, c = 'gray', s = 10) # data
plt.plot(xt, s(xt), linestyle = 'dashed', c = 'red') # regular spline (defaults to 'not-a-knot')
plt.plot(xt, ns(xt), c = 'black') # natural spline
plt.savefig('natural.png')
