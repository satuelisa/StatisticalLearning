from scipy.interpolate import CubicSpline
from matplotlib.pyplot import figure
from numpy.random import normal
import matplotlib.pyplot as plt
import numpy as np

figure(figsize = (6, 4), dpi = 100)

xt = np.arange(-3, 3, 0.1) # true x coordinates
n = len(xt)
magnitude = 0.05
x = np.sort(xt + normal(size = n, scale = magnitude)) # add noise and SORT
yt = np.cos(xt) + xt # true y coordinates
y = yt + normal(size = n, scale = magnitude) # add noise
st = CubicSpline(xt, yt) # fit a cubic spline to the pure data
s = CubicSpline(x, y) # fit a cubic spline to the noisy data

plt.rcParams.update({'text.color': 'green',
                     'xtick.color': 'green',
                     'ytick.color': 'green',
                     'axes.labelcolor': 'green',
                     'axes.edgecolor': 'green',
                     'axes.facecolor':  'none' })
plt.scatter(x, y, c = 'red', s = 10) # data
plt.plot(xt, yt, c = 'gray', linestyle = 'dashed') # pure model
plt.plot(x, st(x), c = 'orange', linewidth = 2, alpha = 0.5) # clean spline
plt.plot(xt, s(xt), c = 'blue', linewidth = 1, alpha = 0.5) # noisy spline
plt.savefig('spline.png', transparent = True)
