from math import sqrt
from numpy import argmax
from random import random, randint

n = 5 # number of features
# possible labels 1, 2, 3
low = 1
high = 3

# generate 'labelled' data at random
N = 100 # amount of known data points
# a list of pairs (x, y)
known = []
for d in range(N):
    known.append(([random() for i in range(n)], randint(low, high)))

def dist(x1, x2): # using euclidean distance for simplicity
    return sqrt(sum([(xi - xj)**2 for (xi, xj) in zip(x1, x2)]))

x = [random() for i in range(n)] # our 'unknown' x

k = 3 # using three nearest neighbors
nd = [float('inf')] * k # their distances (infinite at first)
ny = [None] * k # the labels of the nearest neighbors

for (xn, yn) in known:
    d = dist(x, xn)
    i = argmax(nd)
    if d < nd[i]: # if smaller than the largest of the stored
        nd[i] = d # replace that distance
        ny[i] = yn # and that label

y = sum(ny) / k
xs = ' ' .join(['%.2f' % round(xi ,2) for xi in x])
print(f'[{xs}] is {round(y)}') # round to the closest label (a simple choice)


