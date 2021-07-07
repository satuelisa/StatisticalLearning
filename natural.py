from scipy.interpolate import CubicSpline
from matplotlib.pyplot import figure
from numpy.random import normal
import matplotlib.pyplot as plt
import numpy as np

figure(figsize = (6, 4), dpi = 100)

xt = np.arange(2, 4, 0.01) # true x coordinates
n = len(xt)
magnitude = 0.01
x = np.sort(xt + normal(size = n, scale = magnitude)) # add noise and SORT
y = np.cos(xt) + normal(size = n, scale = magnitude) 
s = CubicSpline(x, y) # as before
ns = CubicSpline(x, y, bc_type = 'natural') # fit a NATURAL cubic spline

plt.rcParams.update({'text.color': 'green',
                     'xtick.color': 'green',
                     'ytick.color': 'green',
                     'axes.labelcolor': 'green',
                     'axes.edgecolor': 'green',
                     'axes.facecolor':  'none' })
plt.scatter(x, y, c = 'gray', s = 10) # data
plt.plot(xt, s(xt), linestyle = 'dashed', c = 'red', alpha = 0.5) # regular spline (defaults to 'not-a-knot')
plt.plot(xt, ns(xt), c = 'orange', alpha = 0.5) # natural spline
plt.savefig('natural.png', transparent = True)
