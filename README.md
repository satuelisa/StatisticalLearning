# Statistical Learning

These are my class notes for
students
[my automated learning graduate course](https://elisa.dyndns-web.com/teaching/sys/apraut/) based
on the
textbook
[The Elements of Statistical Learning](https://link.springer.com/book/10.1007/978-0-387-84858-7) available
in PDF
at
[Haestie's website at Stanford](https://web.stanford.edu/~hastie/Papers/ESLII.pdf).
The data sets used in the examples are from
the [book's website](https://www-stat.stanford.edu/ElemStatLearn).  I
hope that there will be native LaTeX rendering in the markdown soon,
so I could include equations (the present solutions of generating
images are quite ugly in darfk mode). The book uses R in their example
code, so I am going to replant everything in Python for an enriched
experience and more pain on my part.

## Weekly schedule

+ [Chapter 1: Introduction](#introduction)
  * [Homework 1](#homework-1)
+ [Chapter 2: Supervised learning](#supervised-learning)
  * [Section 2.2: Least squares for linear models](#least-squares-for-linear-models)
  * [Section 2.3: Nearest neighbors](#nearest-neighbors)
  * [Homework 2](#homework-2)
+ [Chapter 3: Linear regression](#linear-regression)
  * [Homework 3](#homework-3)
+ [Chapter 4: Linear methods for classification](#classification)
  * [Homework 4](#homework-4)

## Introduction

Frequent uses: prediction, classification, factor identification

Elements: 

+ an outcome measurement (qualitative or quantitative)
+ a set of features (extracted from the input data)
+ a subset of the data set aside for training (the rest for
  validation)
+ the goal is to build a _model_ (a.k.a. learner) 

When the training data comes with a label indicating the intended
outcome, this this **supervised learning**. In the unsupervised case,
there are only features and the goal is more like _clustering_.

Examples: 

- is an email spam?
- does a patient have cancer?
- which handwritten digit is this?
- are some of these genes related to certain types of cancer?

### Homework 1

Identify one or more learning problems in your thesis work and
identify their goals and elements. Write the description
with
[GitHub markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet) in
the README.md of your course repository. If your thesis work has none,
discuss potential project topics with the teacher in on the course
channel in Discord.

## Supervised learning

It is important to mark variables as _categorical_ even if they are
numbers (like the handwritten digits) when the value is not to be
interpreter as an ordering of sorts.

Predicting a **quantitative** variable is _regression_, whereas
predicting a **qualitative** one is _classification_.

An _ordered_ categorical variable (small / medium / large) has a
non-metric ordering.

Notation:

+ input variable (often a vector containing `p` features) `X`
+ quantitative output `Y`
+ qualitative output `G`
+ quantitative
  prediction `Yp`
+ qualitative
prediction `Gp`
+ `n` tuples of quantitative training data `(xi, yi)`
+ `n` tuples of qualitative training data `(xi, gi)`

### Least squares for linear models

The output is predicted as a weighted linear combination of the input
vector plus a constant, where the weights and the constant need to be
learned from the data. 

We can avoid dealing with the constant separately by adding a unit
input
```python
import numpy as np
from numpy.random import uniform, normal

x = np.array([1, 2, 3, 4]).T # a column vector n x 1                                                  
n = len(x) # (n - 1) features plus the constant gives n                                               
w = np.array(uniform(size = n)).T # weights (random for now, between 0 and 1 for starters)            
iyp = np.inner(x, w) # inner product of two arrays        
```
and the **quality** of the prediction is compared as a sum of squares
between the desired values `y` and the predicted values `yp`. 
```python
xt = [ x.T ] # transpose into a row vector, 1 x n                                                     
yp = np.matmul(xt, w) # a scalar prediction, (1 x n) x (n x 1) = 1 x 1                                
assert iyp == yp # should coincide with the inner product from above            
``` 
Lets use matrices to make this more compact:

+ `X` is a matrix where each _column_ is an input vector (`n` inputs)
  and each _row_ is a feature (`p` features)
+ `y` is a vector of the intended labels/outputs (the first element for the
  first column of `X`, the second for the second column, etc.)
+ `yp` is then obtained as `X` multiplying the weight vector `w`

```python
X = np.array([[1, 1, 1],
              [3, 5, 7],
              [8, 7, 3],
              [2, 4, 6]]) # n x p        
assert n == np.shape(X)[0] # features as rows                                                         
p = np.shape(X)[1] # inputs as columns                                                                

# let assume the model is 3 x1 - 2 x2 + 4 x3 - 5 with small gaussian noise                            
def gen(x): # generate integer labels from an arbitrary model                                         
    label = 5 * x[0] + 3 * x[1] - 2 * x[2] + 4 * x[3] \
                  + normal(loc = 0, scale = 0.2, size = 1)
    label = round(label[0])
    print(x, 'gets', label)
    return label

y = np.apply_along_axis(gen, axis = 0, arr = X) # 1 x p, one per input                             

def rss(X, y, w):
    Xt = X.T
    yp = np.matmul(Xt, w) # predictions for all inputs                                                
    yyp = y - yp
    assert np.shape(yp) == np.shape(y)
    return np.matmul(yyp.T, yyp)
```

The best model in this sense is the one that minimizes RSS
```python
for r in range(10): # replicas                                                                        
    wr = np.array(uniform(low = -6, high = 6, size = n)).T
    print(f'{rss(X, y, wr)} for {wr}') # the smaller the better   
``` 
and this is very similar to what the perceptron does (cf. the last
homework of the simulation course, if you took that one already). This
code (which is not a lot) is available in the
file
[`linear.py`](https://github.com/satuelisa/StatisticalLearning/blob/main/linear.py) and
I really recommend sticking to the NumPy routines for creating vectors
and matrices as well as multiplying them. Remember to pay close
attention to the dimensions.

### Nearest neighbors

This technique supposes that we have a set of pre-labelled inputs in
some (metric) space in which we can compute a distance from one input
to another. Then, when we encounter a non-labelled input `x`, we
simply take the `k` nearest labelled inputs and average over their
values of `y` to obtain a `yp` for our `x`. When using just one
neighbor, this effectively reduces to Voronoi cells (cf. the fourth
homework of simulation).

In terms of code, let's generate random data again (the book has a
two-dimensional example):
```python
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
```
For the metric, we will use the root of sum of squares of coordinate
differences using the features as an Euclidean space:

```python
def dist(x1, x2): # using euclidean distance for simplicity
    return sqrt(sum([(xi - xj)**2 for (xi, xj) in zip(x1, x2)]))
```

Now we can find for a given `x` the `k` known data points nearest to
it and store the corresponding labels `y`; we have to consider all of
the labelled data by turn and remember which `k` were the closest
ones, which I will do in a simple for loop for clarity instead of
attempting to iterate over a matrix.

```python
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
```

One the `k` closest labels are known, we average them and I will also
round them as the labels in the known data were integers (a majority
vote would be a reasonable option to rounding, as well as using the median):

```python
y = sum(ny) / k
xs = ' ' .join(['%.2f' % round(xi ,2) for xi in x])
print(f'[{xs}] is {round(y)}') # round to the closest label (a simple choice)
```

The complete code resides
at
[`knn.py`](https://github.com/satuelisa/StatisticalLearning/blob/main/knn.py). 

Things to consider when applying KNN are:

+ which metric to use for the distance
+ should the features be normalized somehow before doing any of this
+ how many neighbors should we use
+ how should be combine the labels of the neighbors to obtain a
  prediction

Read Sections 2.3 and 2.4 to improve your understanding of _why_ and
_how_ these methods are expected to be useful, as well as Section 2.5
to understand why they tend to fall apart when the number of features
is very high (this is similar to what happens in the first simulation
homework for the high-dimensional Brownian motion that no longer
returns to the origin and the eleventh simulation homework with Pareto
fronts where a high number of objective functions renders the
filtering power of non-dominance effectively null and void). The
remainder of the chapter introduces numerous theoretical concepts that
we are likely to stumble upon later on, so please give them at least a
cursorial look at this point so you know that they are there.

### Homework 2

First carry out Exercise 2.8 of the textbook with their ZIP-code data and then
**replicate the process** the best you manage to some data from you own problem that
was established in the first homework.


## Linear regression

We assume the expected value of `y` given `x` to be a linear function
of the features. We can either use the raw features or apply
_transformations_ (roots, logarithms, powers) to the quantitative
ones, whereas the qualitative ones can be numerically coded (varying
the ordering), and we could create _interactions_ by introducing
auxiliary functions to create a "combination feature" that takes as
inputs two or more of the original features. As noted befare, we
usually aim to choose the weights so as to minimize the residual sum
of squares (RSS). Such minimization, theoretically, is achieved by
setting its first derivative to zero and solving for the weights.

```python
def rss(X, y, w):
    yp = np.matmul(X, w) # n predictions, one per input                                                                                
    yyp = y - yp
    return np.matmul(yyp.T, yyp)
```

Let's use more inputs and features this time around and a bit more
noise in the model that generates the labels:

```python
Xt = np.array([[1, 1, 1 , 1, 1, 1, 1, 1], # the constants
               [2, 5, 7, 3, 5, 2, 1, 2], # feature 1
               [8, 6, 3, 1, 9, 4, 3, 2], # feature 2
               [2, 3, 5, 2, 4, 5, 7, 3], # feature 3
               [3, 4, 5, 4, 4, 8, 8, 2]]) # feature 4
X = Xt.T
n = np.shape(X)[0]
p = np.shape(X)[1]
print(p, 'features (incl. a constant)')
print(n, 'inputs')
coeff = uniform(size = p, low = -5, high = 5) # a model

def gen(x): # a bit more randomness this time and use integers
    count = np.shape(x)[0]
    noise = normal(loc = 0, scale = 0.1, size = count).tolist()
    return sum([coeff[i] * x[i] + noise[i] for i in range(count)]) 

y = np.asarray(np.apply_along_axis(gen, axis = 1, arr = X)) 
print(f'{n} labels: {y}')    
```

Now we can replicate the equations for the unique zero of the
derivative:
```python
X = Xt.T
XtX = np.matmul(Xt, X)
XtXi = inv(XtX)
XXX = np.matmul(XtXi, Xt)
w = np.matmul(XXX, y)
ba = rss(X, y, w)
print(f'best analytical {ba:.3f} with weights {w}:')
for (wp, wt) in zip(w, coeff):
    print(f'{wp:.3f} (recovered) for {wt:.3f} (inserted)')
```

We can use it to construct a prediction:

```python
pred = np.matmul(X, w)
print('predictions:')
for (yl, yp) in zip(y, pred):
    print(f'{yp:.3f} for {yl:.3f}')
```

As before, we can also compare it with how just choosing weights at
random and picking the lowest RSS would perform:

```python
lowest = float('inf')
for r in range(1000): # try a bunch of random ones
    wr = np.array(uniform(low = min(w), high = max(w), size = p))
    lowest = min(lowest, rss(X, y, wr))
print(f'the best random attempt has RSS {lowest:.2f} whereas the optimal has {ba:.2f}')
```

Note that this will not work if any of the inputs are perfectly
correlated (as it would result in a singular matrix for
`np.matmul(X.T, T)`, so some pre-processing may be necessary. If `p`
is much larger than `n`, problems may also arise and some
dimensinality-reduction in the feature space could come in handy.  All
of the above takes place
in
[`regression.py`](https://github.com/satuelisa/StatisticalLearning/blob/main/regression.py).

Let's make another version that actually examines the correlations and
attempts to assess the statistical significance of the model, whenever
one manages to be made. First we set some parameters:

```python
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
print('f{n} inputs, {p} features')
dof = n - p - 1 # degrees of freedom
coeff = uniform(size = p, low = -5, high = 5) # coefficients of features (model)
coeff[p // 2] = 0 # make the middle one irrelevant
constants = np.ones((1, n))

def gen(x): # a bit more randomness this time and use integers
    count = np.shape(x)[0]
    noise = normal(loc = 0, scale = 0.1, size = count).tolist()
    return sum([coeff[i] * x[i] + noise[i] for i in range(count)]) 
```

Now we can loop until success (see [`zscore.py`](https://github.com/satuelisa/StatisticalLearning/blob/main/zscore.py)
for the whole thing). We randomly create some data and use the
generation model to assign labels if the correlation checks pass:

```python
features = uniform(low = -50, high = 50, size = (p - 1, n))
Xt = np.vstack((constants, features)) # put the constants on the top row
X = Xt.T # put the inputs in columnns
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
```

Once all of this plays out, we can try to run the math:

```python
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
    dy = y - yp
    dsq = np.inner(dy, dy) # sum([i**2 for i in dy])
    var = dsq / dof # variance
    sd = sqrt(var) # standard deviation
    assert min(v) > 0 # all the vj are positive (that is, the math worked as intended)
```

Usually, one assumes that the deviations between the predicted `yp`
and the expected `y` are normally distributed. If so, the estimated
weights `w` are normally distibuted around the true weights and their
variance is from a chi-squared distribution (`n - p - 1` DoF),
allowing for the calculation of confidence intervals.  To
statistically check if a specific weight in `w` is zero (that is, the
corresponding input in `X` has no effect on the output `y`), we
compute it's Z-score (or F-score as feature-exclusion):

```python
sqv = sqrt(v[j])
ss = sd * sqv            
z = w[j] / ss
excl = np.copy(w)
excl[j] = 0 # constrain this one to zero
rss0 = rss(X, y, excl) # a model without this feature
f = (rss0 - rss1) / rd # just one parameter was excluded
p = chi2.sf(f, dof) # survival function (assuming f == z) 
signif = p < alpha
print(f'\nCoefficient {coeff[j]:.2f} was estimated as {w[j]:.2f}',
      'which is significant' if signif else 'but it is insignificant',
      f'with a p-value of {p:.5f}')
```

We can use F-scores to compare between models with different subsets
of features. We can also add confidence interval calculations with the
normality hypothesis by adding a bit of calculations:

```python
width = z1a * sd
low = w[j] - width
high = w[j] + width
print(f'with a confidence interval [{low}, {high}]', 
      'that contains zero' if low < 0 and high > 0 else '')
```

If we have **multiple** outputs, meaning that  `Y` is a matrix,
too. This happens in [`multreg.py`](https://github.com/satuelisa/StatisticalLearning/blob/main/multreg.py)
using the math from Section 3.2.3 using `from numpy.linalg import qr, inv` for the QR-decomposition.

First, we need to make the generator create multiple outputs per input:

```python
k = 3 
print(f'{n} inputs, {p} features')
coeff = uniform(size = (k, p), low = -5, high = 5) # coefficients for the k output

def gen(x): # a bit more randomness this time and use integers
    count = np.shape(x)[0]
    noise = normal(loc = 0, scale = 0.1, size = count).tolist()
    return [sum([coeff[j, i] * x[i] + noise[i] for i in range(count)]) for j in range(k)]
```
Then we generate the labels and throw in the math:

```python
y = np.asarray(np.apply_along_axis(gen, axis = 1, arr = X))
Q, R = qr(X)
Ri = inv(R)
Qt = Q.T
RiQt = np.matmul(Ri, Qt)
w = np.matmul(RiQt, y)
for i in range(k):    
    for j in range(p):
        print(f'Coefficient {coeff[i, j]:.2f} of output {i + 1} was estimated as {w[j, i]:.2f}')
```

We can also make predictions:

```python
QQt = np.matmul(Q, Qt)
yp = np.matmul(QQt, y)
for i in range(n):
    print(f'Expected {y[i, :]}, predicted {yp[i, :]}')
```

Please read with careful thought the sections on subset-selection
methods, as it is important to stick to the smallest reasonable model
instead of simply shoving in every feature and interaction one can
think of. I know it's a lot of text, but this is stuff you need to be
familiar with, although in practice you will usually employ an
existing library for all of this. For python code on the selection
methods, check out the thorough examples by
[Oleszak](https://towardsdatascience.com/a-comparison-of-shrinkage-and-selection-methods-for-linear-regression-ee4dd3a71f16).

### Homework 3

Repeat the steps of the prostate cancer example in Section 3.2.1 using
Python, first as a univariate problem using the book's data set and
then as a multi-variate problem with data from your own
project. Calculate also the p-values and the confidence intervals for
the model's coefficients for the uni-variate version. Experiment,
using libraries, also with subset selection.

## Classification

Now the predictor takes discrete values (each input gets assigned to a
class). The task is to divide the input space into regions that
correspond to each class. In a linear case, the class boundaries are
hyperplanes.

When using regression, we can compute the weights `w` as before and
then obtain posterior probabilities. The whole thing is at 
[`postprob.py`](https://github.com/satuelisa/StatisticalLearning/blob/main/postprob.py):

First we make a model with integer labels 1 and 2.
```python
import numpy as np  
from math import exp
from numpy.linalg import qr, inv 
from numpy.random import normal

np.set_printoptions(precision = 0, suppress = True)
Xt = np.array([[1, 1, 1, 1, 1, 1, 1, 1], # constant
               [2, 5, 7, 3, 5, 2, 1, 2], # feature 1
               [8, 6, 3, 1, 9, 4, 3, 2], # feature 2
               [2, 3, 5, 2, 4, 5, 7, 3], # feature 3
               [3, 4, 5, 4, 4, 8, 8, 2]]) # feature 4
X = Xt.T
coeff = np.array([0, 1, 2, 3, 4]) # zero weight for the constant
threshold = 30 # make two classes

def gen(x): # a bit more randomness this time and use integers
    count = np.shape(x)[0]
    noise = normal(loc = 0, scale = 0.1, size = count).tolist()
    value = sum([coeff[i] * x[i] + noise[i] for i in range(count)])
    return (value > threshold) + 1 # class one or class two

intended = np.asarray(np.apply_along_axis(gen, axis = 1, arr = X))
print('Intended labels')
print(intended)
```

Then we make indicator vectors (true versus false) for each of the two classes:

```python
y = np.array([(1.0 * (intended == 1)), (1.0 * (intended == 2))]).T
```

With those, we recycle the multiregression math with the QR decomposition:

```python
Q, R = qr(X)
Ri = inv(R)
Qt = Q.T
RiQt = np.matmul(Ri, Qt)
w = np.matmul(RiQt, y)
```

We can also make and round predictions directly as before:

```python
QQt = np.matmul(Q, Qt)
yp = np.matmul(QQt, y)
print('Assigned indicators')
print(y.T)
print('Predicted indicators (rounded)')
print(np.fabs(np.round(yp, decimals = 0)).T)
```

but the new math allows us to compute for each class the probability that the input belongs to it:

```python
wt = w.T
for c in [1, 2]: # for both classes
    b = wt[c - 1]
    print('Class', c)
    for i in range(len(intended)):
        x = X[i]
        yd = intended[i]
        eb = exp(np.matmul(b, x)) 
        ed = 1 + eb
        pThis = eb / ed
        pOther = 1 / ed
        ok = pThis > pOther
        print(f'{x} should be {yd}: this {pThis:.2f} vs. other {pOther:.2f} ->', 'right' if ok else 'wrong')
```

Most of the input are correctly labelelled regardless of which class
is used to compute the posterior probabilities. As the book points
out, in practice, class boundaries have no particular reason to be
linear. 

We can also predict without explicitly computing the weights as in Equation (4.3) of the book
[`indmat.py`](https://github.com/satuelisa/StatisticalLearning/blob/main/indmat.py), using the same `X`

```python
y = np.asarray(np.apply_along_axis(gen, axis = 1, arr = X))
Y = np.array([(1.0 * (y == 1)), (1.0 * (y == 2))]).T
XtXi = inv(np.matmul(Xt, X))
XXtXi = np.matmul(X, XtXi)
XXtXiXt = np.matmul(XXtXi, Xt)
Yp = np.matmul(XXtXiXt, Y)
print('Assigned indicators')
print(Y.T)
print('Predicted indicators')
print(np.fabs(Yp.T)) # the -0 bother me so I use fabs
```

If the classes share a covariance matrix, we can apply _linear
discriminant analysis_ (LDA) to compute linear discriminant functions
to obtain the class boundaries. For two classes, this is the same as
using the above calculations with an indicator matrix, but not in the
general case. When the covariance matrix cannot be assumed to be the
same, _quadratic discriminant functions_ arise. An option to LDA/QDA
is _regularized discriminant analysis_ (RDA). Subspace projections may
also come in handy.

**Logistic regression** for classes `1, 2, ..., k` refers to modelling
logit transformations (logarithm of the posterior probability of a
specific class `i` normalized by the posterior probability of the last
class `k`) with linear functions in the input space, using a linear
function `ci + bi.T x` where `ci` is the constant term and `bi` the
weight vector for class `i`, with such an equation for each `i < k`
and then iteratively solving for the values of the `ci` and the
vectors `bi`. This may or may not converge and reles on an initial
guess for the weights. A regularized version exists for this as well.

Please read all of Chapter 4 carefully. Also perceptrons (already
mentioned before, the ones we worked with in the simulation class) are
discussed in Section 4.5.1 in the context of hyperplane separation
(ideally, the decision boundary would be as separated from the
training inputs as it can).

### Homework 4

Pick one of the examples of the chapter that use the data of the book
and replicate it in Python. Then, apply the steps in your own data. 
