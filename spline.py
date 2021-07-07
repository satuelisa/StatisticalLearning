from scipy.interpolate import CubicSpline
from numpy.random import normal
import matplotlib.pyplot as plt
import numpy as np

xt = np.arange(-7, 7, 0.05) # true x coordinates
n = len(xt)
magnitude = 0.05
x = np.sort(xt + normal(size = n, scale = magnitude)) # add noise and SORT
yt = np.cos(xt) + xt # true y coordinates
y = yt + normal(size = n, scale = magnitude) # add noise
st = CubicSpline(xt, yt) # fit a cubic spline to the pure data
s = CubicSpline(x, y) # fit a cubic spline to the noisy data

plt.rcParams.update({'text.color': 'green',
                     'axes.labelcolor': 'green',
                     'axes.facecolor':  'none' })
plt.scatter(x, y, c = 'red') # data
plt.plot(xt, yt, c = 'blue', linestyle = 'dashed') # pure model
plt.plot(x, st(x), c = 'green', linewidth = 2) # clean spline
plt.plot(xt, s(xt), c = 'black', linewidth = 1) # noisy spline
plt.savefig('spline.png', transparent = True)
