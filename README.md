# Statistical Learning

These are my class notes
for
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
images are quite ugly in dark mode). The book uses R in their example
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
+ [Chapter 5: Basis expansions](#basis-expansions)
  * [Homework 5](#homework-5)
+ [Chapter 6: Smoothing](#smoothing)
  * [Homework 6](#homework-6)
+ [Chapter 7: Assessment](#assessment)
  * [Homework 7](#homework-7)
+ [Chapter 8: Inference](#inference)
  * [Homework 8](#homework-8)
+ [Chapter 9: Additive models and trees](#additive-models-and-trees)
  * [Homework 9](#homework-9)
+ [Chapter 10: Boosting](#boosting)
  * [Homework 10](#homework-10)
+ [Chapter 11: Neural networks](#neural-networks)
  * [Homework 11](#homework-11)
+ [Chapter 12: SVM and generalized LDA](#SVM-and-generalized-LDA)
  * [Homework 12](#homework-12)
+ [Chapter 13: Prototypes and neighbors](#prototypes-and-neighbors)
  * [Homework 13](#homework-13)
+ [Chapter 14: Unsupervised learning](#unsupervised-learning)
	* [Homework 14](#homework-14)
+ [Chapter 15: Random forests](#random-forests)
  * [Homework 15](#homework-15)
+ [Chapter 16: Ensemble learning](#ensemble-learning)
  * [Homework 16](#homework-16)
+ [Chapter 17: Graphs](#graphs)
  * [Homework 17](#homework-17)
+ [Chapter 18: High dimensionality](#high-dimensionality)
  * [Final project](#final-project)

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

# let assume the model is 3 x1 - 2 x2 + 4 x3 - 5 with small Gaussian noise                            
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

and this is very similar to what the perceptron does (cf. the [last
homework](https://elisa.dyndns-web.com/teaching/comp/par/p12.html) of
the [simulation
course](https://elisa.dyndns-web.com/teaching/comp/par/), if you took
that one already). This code (which is not a lot) is available in the
file
[`linear.py`](https://github.com/satuelisa/StatisticalLearning/blob/main/linear.py)
and I really recommend sticking to the NumPy routines for creating
vectors and matrices as well as multiplying them. Remember to pay
close attention to the dimensions.

### Nearest neighbors

This technique supposes that we have a set of pre-labeled inputs in
some (metric) space in which we can compute a distance from one input
to another. Then, when we encounter a non-labeled input `x`, we
simply take the `k` nearest labeled inputs and average over their
values of `y` to obtain a `yp` for our `x`. When using just one
neighbor, this effectively reduces to Voronoi cells (cf. the [fourth
homework of
simulation](https://elisa.dyndns-web.com/teaching/comp/par/p4.html)).

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

# generate 'labeled' data at random
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
the labeled data by turn and remember which `k` were the closest
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
is very high (this is similar to what happens in the [first simulation
homework](https://elisa.dyndns-web.com/teaching/comp/par/p1.html) for
the high-dimensional Brownian motion that no longer returns to the
origin and the [eleventh simulation
homework](https://elisa.dyndns-web.com/teaching/comp/par/p11.html)
with Pareto fronts where a high number of objective functions renders
the filtering power of non-dominance effectively null and void). The
remainder of the chapter introduces numerous theoretical concepts that
we are likely to stumble upon later on, so please give them at least a
cursory look at this point so you know that they are there.

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
inputs two or more of the original features. As noted before, we
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
dimensionality-reduction in the feature space could come in handy.  All
of the above takes place
in
[`regression.NY`](https://github.com/satuelisa/StatisticalLearning/blob/main/regression.py).

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
z1a = norm.ppf(1 - alpha) # Gaussian percentile 

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
weights `w` are normally distributed around the true weights and their
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
Python, first as a uni-variate problem using the book's data set and
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

With those, we recycle the multi-regression math with the QR decomposition:

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

Most of the input are correctly labeled regardless of which class is
used to compute the posterior probabilities. As the book points out,
in practice, class boundaries have no particular reason to be linear.

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

**Logistic regression** for classes `1, 2, ..., k` refers to modeling
logit transformations (logarithm of the posterior probability of a
specific class `i` normalized by the posterior probability of the last
class `k`) with linear functions in the input space, using a linear
function `ci + bi.T x` where `ci` is the constant term and `bi` the
weight vector for class `i`, with such an equation for each `i < k`
and then iteratively solving for the values of the `ci` and the
vectors `bi`. This may or may not converge and relies on an initial
guess for the weights. A regularized version exists for this as well.

Please read all of Chapter 4 carefully. Also perceptrons (already
mentioned before, the ones we worked with in the simulation class) are
discussed in Section 4.5.1 in the context of hyperplane separation
(ideally, the decision boundary would be as separated from the
training inputs as it can).

### Homework 4

Pick one of the examples of the chapter that use the data of the book
and replicate it in Python. Then, apply the steps in your own data. 

## Basis expansions

It is intuitively clear that real-world relationships between factors
in data are unlikely to be linear. One work-around that is compatible
with the math we have used thus far is including new features that are
transformations of the original ones (or their combinations). This, of
course, makes the feature-selection and model-minimization algorithms
even more relevant as we are now blowing the number of possible
features sky-high.

The transformations could be specific functions (attempted with a
Tukey ladder, for example) such as logarithms or powers, but _basis
expansions_ is a more flexible way to get this done. This chapter
deals with using piece-wise polynomials and splines as also touches the
use of wavelets.


There is a toy example of how to work with transformations
in[`transform.py`](https://github.com/satuelisa/StatisticalLearning/blob/main/transform.py).

It is common to doing this assuming **additivity**: each _model_
feature is a linear combination of basis functions applied to the
_raw_ features, among a set of `p` basis functions. Basis functions
that do not result in a significant contribution to the model can be
discarded with selection methods, and also the coefficients of the
linear combination could be restricted (this is called
_regularization_).

In a piecewise-linear function, the book refers to the points at which
the pieces change as _knots_ (do not confuse this with the
mathematical concept of a knot). **Splines** of order `m` are
piecewise-polynomials with `m - 2` continuous derivatives; cubic (`m =
3`) splines often result in polynomials where the knot locations are
no longer easy to spot to the naked eye. If the _locations_ of the
knots are fixed (instead of flexible), these are called "regression
splines". Play with the example
in
[`spline.py`](https://github.com/satuelisa/StatisticalLearning/blob/main/spline.py) for
a bit, varying the magnitude of the noise in the calls to `normal` and
the step size (the third parameter of `np.arange(-7, 7, 0.05)`) to see
how the spline behaves. Also mess with the function itself. We use
`from scipy.interpolate import CubicSpline` to create the spline.
Note that `CubicSpline` wants an ordered `x`, so we  **have to** sort
after we add the noise. 

![The resulting splines](https://github.com/satuelisa/StatisticalLearning/blob/main/spline.png)

The figure shows the data with red
dots, a green dashed line for the function we used, and two splines:
one fitted to the "model" and another one fitted to the noisy version
of the coordinates.

```python
plt.scatter(x, y, c = 'red') # data
plt.plot(xt, yt, c = 'gray, linestyle = 'dashed') # pure model
plt.plot(x, s(x), c = 'orange) # clean spline
plt.plot(xt, s(xt), c = 'blue) # noisy spline
```

Undesirable behavior at boundaries can be somewhat tamed by using
_natural_ splines 
([`natural.py`](https://github.com/satuelisa/StatisticalLearning/blob/main/natural.py)):
```python
s = CubicSpline(x, y) # as before, the default is 'not-a-knot'
ns = CubicSpline(x, y, bc_type = 'natural') # fit a NATURAL cubic spline

plt.scatter(x, y, c = 'gray', s = 10) # data now in GRAY with small dots
plt.plot(xt, s(xt), linestyle = 'dashed', c = 'red') # the default one in RED
plt.plot(xt, ns(xt), c = 'orange') # natural spline now in ORANGE
```

If there is a difference, it appears at the edges of the plot. Rerun
until you've seen it at both ends (the random noise affects the spline
computations so the outcome differs at each execution).

![Natural splines](https://github.com/satuelisa/StatisticalLearning/blob/main/natural.png)

There are many fine details to applying this, so it is important to
read all of Chapter 5. 

An alternative family of basis functions are _wavelets_ that are
orthonormal. There is
a [library for that](https://pywavelets.readthedocs.io/en/latest/).

### Homework 5

Fit splines into single features in your project data. Explore options
to obtain the best fit.

## Smoothing

Instead of fitting one model to all of the data, we can fit simple,
local models for each point we are interested in; the book calls these
_query points_. The way "locality" is defined is by assigning a
function that sets a weight for each data point based on its distance
from the query point. These weight-assignment functions are called
**kernels** and are usually parameterized in a way that controls how
wide the neighborhood will be. Kernels are the topic of Chapter 12.

This is like the simple thing we did for the nearest-neighbor
averaging method in
[`knn.py`](https://github.com/satuelisa/StatisticalLearning/blob/main/knn.py)
in Chapter 2. To illustrate the downside of the simple procedure,
let's use KNN to estimate a curve like the ones we used for the spline
examples in the previous chapter; this code is in
[`knn2.py`](https://github.com/satuelisa/StatisticalLearning/blob/main/knn2.py). First
we create data and pick some query points at random:

```python
low = -7
high = 12
xt = np.arange(low, high, 0.1) 
n = len(xt)
m = 0.1
x = np.sort(xt + normal(size = n, scale = m)) 
y = np.cos(xt) - 0.2 * xt + normal(size = n, scale = m) # add noise
xq = np.sort(uniform(low, high, size = n // 2)) # sorted
```

Then we pick the `k` closest data points (in terms of `x` as we do not
know the `y`, since we are trying to _estimate_ it) and use their
average as the estimate for the corresponding `y`:

```python
yq = []
k = 10
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
```

![KNN](https://github.com/satuelisa/StatisticalLearning/blob/main/knn2.png)

The curve resulting by connecting the query point estimates is, of
course, generally discontinuous, as expected. Throwing in some more
math, we can patch the existing math for a smoother result. First we
define the auxiliary function and the kernel itself:

```python
def dt(t): # Eq. (6.4)
    return (3/4) * (1 - t**2) if fabs(t) <= 1 else 0

def kernel(qp, dp, lmd = 0.2): # Eq. (6.3)
    return dt(fabs(qp - dp) / lmd)
```

and then we use these to calculate the `y` estimates as weighted averages 
(the whole thing is at
[`kernel.py`](https://github.com/satuelisa/StatisticalLearning/blob/main/kernel.py):

```python
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
```

![Smoothing](https://github.com/satuelisa/StatisticalLearning/blob/main/kernel.png)

Of course we could use another `dt` like that of Eq. (6.7) or a
different kernel. Now, applying this same idea to linear or even
polynomial regression, we could also perform that in a local
neighborhood, using a kernel to assign weights to the known data
points within that local neighborhood. Much of this is playing with
the parameters (like `lmd` in `kernel`) to obtain the best possible
model.

### Homework 6

You guessed it. That's the homework. Build some local regression model
for your data and adjust the parameters. Remember to read all of
Chapter 6 first to get as many ideas as possible. 

## Assessment

Suppose we have tons of data. We first separate our available data, at
random, into three non-overlapping sets:
+ Training set (about 50 to 70 %)
+ Validation set (about 15 to 25 %)
+ Test set (about 15 to 25 %) and life is sort of easy from there
on. Use the training set to build a model, use the validation set to
adjust any parameters or details that might need tuning, and then
assess the model performance and quality in terms of the test set
without making any tweaks to the model based on those results.

The seventh chapter is about what do when you do **not** in fact have
this luxury and you have to validate analytically or using the same
data repeatedly (like cross-validation and bootstrap).

So, if we have tons of potential models with various different subsets
and transformations of the features, using different techniques, how
do we quantify how good these models are and how would be go about
choosing one in particular to actually use for whatever application we
have in mind.

Let `Y` again be the target variable (i.e., what we are trying to
model) and `X` the inputs (i.e., in terms of what we are building the
model for `Y`). Denote by `f(X)` a prediction model that produces the
estimates for `Y` as a function of `X`. 

A *loss function* `L(Y, f(X))` is a measurement of the prediction
error. Typical examples include the _squared_ error `(Y - f(X))**2`
and the _absolute_ error `np.abs(Y - f(X))`. For categorical data, we
can go with an indicator function that returns zero when the labels
match (intended versus assigned) and one otherwise or a log-likelihood
version such as _deviance_ as in Equation (7,6).

In terms of such a function `L`, we can define _generalization error_
(a.k.a. _test error_) as the expected value of `L`, conditioned on a
(specific, fixed) test set `T`, where the inputs of `T` are _not_ the
ones we included in the training set `X` that was used to build the
model in the first place.  

The expected value of this expectation over possible test sets is the
_expected prediction error_. Remember that when you are dealing with
data instead of theory, averaging over several independent samples is
usually a decent way to estimate an expected value of a quantity. In
effect, if we average the loss over a set of independent training
sets, we get what is called _training error_.

If you recycle any of the inputs in `X` in `T`, you are likely to see
a lower error, but it does not imply you're "doing better" --- of
course the error is lower if you test with the same data you modeled
on. This is **not** a good thing. You want a low error on _previously
unseen_ data. The difference of errors when measured  
(1) on actual validation or testing data (larger error) and (2) on the
training data itself (smaller error) is called the _optimism_ of the
model. Analytically, optimism increases linearly with the number of
features we use (raw or transformed). If we estimate optimism
(analytically) and then add it to the training error, we get an
estimate for the actual prediction error. This is what is discussed in
Section 7.5 of the book.

We assume the error to be a a sum of three factors: an _irreducible
error_ plus _bias_ (squared) plus _variance_, the first of which is
the variability present in the phenomenon that is being modeled, the
second measures how far the estimated mean is from the true mean, and
the third measures how the estimates deviate from their mean. Complex
models tend to achieve a lower bias (a more precise mean) but with the
cost of a higher variance. Section 7.6 discusses how to determine the
"effective number of parameters" for a model, whereas Section 7.8
describes the "minimum description length" which is another approach
for the same issue: how to quantify model complexity. There are
sections on Bayesian approaches as well as heavier stuff, too, for the
mathematically oriented.

The book provides us with analytical expressions for this sum for
linear KNN and ridge regression methods. Tuning a model parameter
results in a bias-variance trade-off: lowering one results in an
increase in the other.

A very popular way to go about prediction error estimation is called
**cross-validation** that estimates, in a direct manner, the expected
error over independent test samples. First, take your data and split it
into `c` non-overlapping subsets. Then, iterate `c` times as follows:
set the `c`th subset aside for validation and use the other `c - 1`
subsets, combined, as training data. Average over the loss functions
to obtain an estimate for the prediction error. Yes, you should try
different values of `c` to find out which one works with any
particular data-model combination. It is important to carry out the
entire process `c` times with the resulting test sets, instead of
building once and then attempting to "only validate" multiple times
(read Section 7.10 to understand why it is bad to limit it to just
some of the steps). A conceptual toy example is available at
[`crossval.py`](https://github.com/satuelisa/StatisticalLearning/blob/main/crossval.py)
with `c = 5` and `n = c * 20` meaning that it uses 80 samples to train
and 20 to validate on each iteration. We use the same math as before
for linear regression to compute predictions and then average over their errors:

```python
e = list() # errors go here
for r in range(c):
    # since they are randomly generated, take every cth column starting at r 
    Xc = X[r::c] 
    yc = y[r::c]
    yp = predict(Xc, yc)
    diff = (yc - yp)
    e.append(np.inner(diff, diff)) # SSQ
print('Iterations', np.array(e))
print('Estimated prediction error', sum(e) / c) # average over them
```
Another common way to go about this is **bootstrap**. Here, we
generate `b` random samples of the same size than the original
training set, but _with replacement_, meaning that the individual
inputs may (and usually will) repeat. We fit the model to these "fake"
data sets and compare the fits, but simply averaging over the losses
here suffers from the fallacy of using the same data for testing and
validation, resulting in "artificially low" error estimates. Read
Section 7.11 for more details on this.

A conceptual toy example for this as well is available at
[`bootstrap.py`](https://github.com/satuelisa/StatisticalLearning/blob/main/bootstrap.py)
with `b = 5` and `n = 100` meaning that it uses 100 samples to train a
base model and then samples from these a set of `n` but _with_
replacement for each of the `b` replicas on each iteration. Note that
the same math we used for linear regression to compute predictions
**will not work** as sampling with replacement makes the matrix
singular so we will use a `sklearn.linear_model` implementation for
`LinearRegression` instead to circumvent this issue. The toy example
does _not_ actually compute any errors or perform validation; it
simply builds the models and spits out the coefficients so you can
stare at them and observe that they do in fact vary. We use `choices`
from `random` to sample with replacement.

```python
baseline = LinearRegression().fit(X, y) 
print(baseline.coef_) 

pos = [i for i in range(n)]
for r in range(b): 
    Xb = np.zeros((n, p))
    yb = np.zeros(n)
    i = 0
    for s in choices(pos, k = n): 
        Xb[i, :] = X[s, :]
        yb[i] = y[s]
    model = LinearRegression().fit(Xb, yb) 
    print(model.coef_) # replica model
```

Also read the last section of the chapter before jumping to the homework.

### Homework 7

You guessed it again, you clever devil: apply both cross-validation
and bootstrap to your project data to study how variable your results
are when you switch the test set around.

## Inference

The methods applied thus far concentrate on minimizing an error
measure such as the sum of squares or _cross entropy_ (check out the
[tutorial by Jason
Brownlee](https://machinelearningmastery.com/cross-entropy-for-machine-learning/)
if the word "entropy" still makes you anxious); now, instead, we
maximize the _likelihood_ in a Bayesian sense.

First, consider doing a bootstrap but by adding Gaussian noise to the
predictions and then use the minimum and maximum values over the
replicas as upper and lower limits of confidence bands to the
estimate. Let's make a wide example plot

```python
from matplotlib.pyplot import figure
figure(figsize = (20, 4), dpi = 100)
```

so that we can draw for each input the bounds we get from a bunch of
bootstrap replicas using their minimums and maximums:

```python
low = [float('inf')] * n
high = [-float('inf')] * n
pos = [i for i in range(n)]
for r in range(b): 
    Xb = np.zeros((n, p))
    yb = np.zeros(n)
    i = 0
    for s in choices(pos, k = n): 
        Xb[i, :] = X[s, :]
        yb[i] = y[s]
    model = LinearRegression().fit(Xb, yb) 
    repl = model.predict(X) # predict 
    for i in range(n):
        prediction = repl[i]
        low[i] = min(low[i], prediction)
        high[i] = max(high[i], prediction)

plt.xlabel('Input index')
plt.ylabel('Predictions')
plt.vlines(pos, low, high, zorder = 1) # behind
plt.scatter(pos, low, c = 'red', zorder = 2) # front
plt.scatter(pos, high, c = 'blue', zorder = 2) # front
```

The whole thing is in
[`bands.py`](https://github.com/satuelisa/StatisticalLearning/blob/main/bands.py)
and the resulting figure shows errorbars-like things.

![Errorbar
thingies](https://github.com/satuelisa/StatisticalLearning/blob/main/bands.png)

With an infinite number of replicas, this would result in the same
bands as a least-squares approach. Averaging over the predictions is
called _bagging_ and is discussed in Section 8.7 with cool
examples. If instead of averaging, we opt for a best-fit approach,
then it's called _bumping_ and Section 8.9 is the place to be.

Suppose that `g` is a probability density function for the
observations, parameterized in some way. For example, a Gaussian
distribution would have two parameters: the mean and the variance.

A _likelihood function_ is the _product_ of the values of `g` under
those parameters for all of our input vectors. Take the logarithm of
that so you can deal with a sum instead of a product, and you have the
corresponding _log-likelihood_ function.

Now, the _score_ of a certain parameter-data combo is the value of its
log-likelihood. We want to make model choices that maximize this
score.

In a Bayesian spirit, one can compute the conditional (posterior)
distribution for those parameter values given the input data which we
can then combine with the conditional probabilities of new data given
the parameters to compute a _predictive distribution_. One would like
to sample this posterior distribution, but in practice it tends to be
such a mess that you will want to go MCMC on this (see the [fifth
homework](https://elisa.dyndns-web.com/teaching/comp/par/p5.html) of
the [simulation
course](https://elisa.dyndns-web.com/teaching/comp/par/) that I keep
mentioning and read Section 8.6 for more info on how that
works). Averaging over Bayesian models is discussed in Section 8.8.

A careful examination of Section 8.4 is a wonderful way to develop a
headache over these concepts. Section 8.5, however, explains how the
_expectation-maximization_ (EM) works. This is delightfully
complemented by the discussion of how to do it in Python given by
[Siwei
Causevic](https://towardsdatascience.com/implement-expectation-maximization-em-algorithm-in-python-from-scratch-f1278d1b9137).

### Homework 8

Yeah, do EM with your data following the from-scratch steps of
[Causevic](https://towardsdatascience.com/implement-expectation-maximization-em-algorithm-in-python-from-scratch-f1278d1b9137).

## Additive models and trees


Still within the realm of supervised learning, we now look at
different options for structuring a regression function. 

In an _additive_ model, we assume that the expectation of `Y` given
`X`is a linear combination of individual functions `f(x)` over the
features that form `X` (again assuming a unit vector there for the
constant term). The relationship between the mean of `Y` conditioned
on `X` to that sum of the `f(x)`terms is given by a **link
function**(options include _logit_, _probit_, identity or just a
logarithm). The task is to come with the those functions `f(x)` for
each feature; they do not need to be all linear or nonlinear
(especially mixing quantitative and qualitative features often
requires diverse `f`s). What is optimized is a (possibly weighted) sum
of squares of the errors (with tons of tunable parameters). Iterative
methods that keep adjusting the functions until they stabilize in a
sense are typical (cf. _backfitting_ in Algorithm 9.1 of the book).

Methods that use _trees_ work by partitioning the feature space into
(non-overlapping) hypercubes and fitting a separate model to each
cube. Such partitioning is often done recursively. The magical part
is, naturally, choosing the partition thresholds for the features,
usually by optimizing some quantity that measures "distance" or
"separation" between the two sides. This is very common for
classification problems and the methods are (quite evidently) called
_classification trees_. Potential performance measures include
counting classification errors, computing the deviance
(cross-entropy), as well as the **Gini index** (check out the tutorial
by [Kimberly
Fessel](http://kimberlyfessel.com/mathematics/applications/gini-use-cases/)
to understand this in general and [Shagufta Tahsildar's blog
post](https://blog.quantinsti.com/gini-index/) for the specific
context of classification).

The _patient rule induction method_ (PRIM) discussed in Section 9.3 is
available as a
[`pip`package](https://github.com/Project-Platypus/PRIM), whereas
_multivariate adaptive regression splines_ (MARS) discussed in Section
9.4 using `scikit-learn`is explained in [Jason Brownee's blog
post](https://machinelearningmastery.com/multivariate-adaptive-regression-splines-mars-in-python/).

If you dislike the idea of splitting the feature space into
non-overlapping sub-regions, _hierarchical mixtures of experts_ (HME)
is a probabilistic variant of the tree-based approach (Section 9.5),

### Homework 9

Read through the spam example used throughout Chapter 9 and make an
effort to replicate the steps for your own data. When something isn't
quite applicable, discuss the reasons behind this. Be sure to read
Sections 9.6 and 9.7 before getting started.

## Boosting 

Much like bagging, **boosting** refers to making use of multiple weak
models (with classification errors only a wee bit better than random
chance, for example) in forming a decently-performing one.

So, produce a total `k` weak classifiers, each with a modified input
set (created for example by weighted sampling, where each iteration
assigns more weight to those samples that the preceding ones
mis-labeled), and then combine these (possibly again with weights) to
produce the final classification. An example of this approach is
[AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
and the idea also works for
[regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html).
The thing to be optimized are the weights of the component models upon
computing the combined model. For AdaBoost, the component weights will
be a logarithm of the success/error ratio (see Algorithm 10.1 in the
book). The tenth chapter, overall, discusses how and why this
works. All of this can be, of course, tweaked in tons of ways, including 

- gradually expanding the set of possible _basis functions_ over the iterations
- using different _loss functions_, such as exponential ones
- using this on _trees_ instead of 'flat' classifiers
- optimizing the loss function with gradient boosting (such as steepest descent)
- _regularization_ to avoid overfitting; _shrinkage_ to keep the models small
- bootstrapping or bagging to average stuff out (and reduce variance)
 
### Homework 10

Replicate the steps of the California housing example of Section
10.14.1 (with some [library
implementation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html))
unless you really want to go all-in with this) to explore potential
dependencies and interactions in the features of your data.

## Neural networks

General idea: use the original features to create linear combinations
of them (I will refer to these building blocks as _neurons_) and then
build a model as a non-linear function of these.

**Projection pursuit regression** (PPR): `f(X) =
sum([g[k](np.matmul(w[k].T, X) for k in neurons])`, meaning that for
each neuron `k` we compute a scalar multiplying the inputs from the
left with the transposed _weights_ of that neuron and use a specific
_ridge function_ `g` for each neuron, then summing over the resulting
values. Both the weights `w` and the functions `g` need to be
estimated with an optimization procedure of sorts. Note that
flexibility in the choice of the `g` allows representing for example
multiplicative features (see Section 11.2 of the book). The larger the
value of `k`, the more complex the model, of course. In economics, the
case `k = 1` is called the _single index model_.

In a neural network, each neuron of the first (hidden)
layer_activates_ as a function of `np.matmul(w.T, X)` (for example a
sigmoidal function), yielding an intermediate input. These `k`
intermediate inputs are fed into an output layer, the neurons of which
produce their own linear combinations of the intermediate inputs,
which are then fed into a function that gives the output of the
network. See Figure 11.2 of the book for an illustration of this
structure. It is common to include the constant unit feature to
represent _bias_. The weights of the neurons are what needs to be
estimated to minimize error (analytically or just iteratively; SSQ for
regression, cross-entropy for classification), avoiding over-fitting
as it ruins generalization to unseen data. **Back-propagation** refers
to computing predictions using current weights, then calculating the
errors, and using these errors to make adjustments to the weights that
gave rise to them. This is pleasant to parallelize. The adjustments
can be either made after each individual input has been processed or
in batch mode.

Frequent issues:

+ good initial weights (random, small)
+ over fitting (use a tuning parameter)
+ normalization of inputs
+ choosing `k`for the hidden layer 
+ deciding whether to also have `k` outputs or fewer
+ deciding whether multiple hidden layers would help
+ local minima

### Homework 11

Go over the steps of the ZIP code examples in Chapter 11 and replicate
as much as you can with your own project data. Don't forget to read
the whole chapter before you start.

## SVM and generalized LDA

Previously, we wanted to separate classes by hyperplanes, but it's not
always the case that the classes _are_ in fact **linearly**
separable. We now discuss methods for creating non-linear boundaries
by working in a transformation of the feature space in such a way that
the boundaries in that transformed space are linear but their
projections, so to speak, to the original feature space might not be.

Lets denote the margin at which the data points are from the
separating hyperplane by `M`, making the width of the separation band
`2M`. As developed in Equation (12.4) of the book, `M = 1 /
np.linalg.norm(w)` where `w` are the weights of the linear model (the
ones the book denotes by beta). 

When the classes are not linearly separable, some of the data will end
up on the wrong side of the boundary, which calls for (positive)
_slack variables_ which I will denote by `s[i]` and the book denotes
by xi. 

For maximizing `M` even with some of the data on the wrong size, we
can restrict to 

`np.matmul(y[i], np.matmul(x[j].T, w[i]) + constant) >= M - s[i]`

which looks easy but results in a non-convex problem and
that's inconvenient. So instead we opt to require 

`np.matmul(y[i], np.matmul(x[j].T, w[i]) + constant) >= M * (1 -
s[i])`

altering the
margin width multiplicatively (`*`) instead of additively (`-`;
a subtraction is just an addition of a negative quantity; we could
also go for `s[i] <= 0`if we wanted to write it with a `+` in both
formulation).

We assume that the sum of the slack variables is limited from above by
a constant which directly limits by how much the predictions can
escape on the wrong side: if `s[i] > 1`, the input `x[i]` is
misclassified.

A SVM (Support Vector Machine) is a classifier that optimizes this
formulation through a Lagrangian dual, but the authors warn that
Section 12.2.1 might be a bit of a pain to process. Kernels are used
to expand the feature space into a very large transformed one and
regularization helps avoid overfitting. One can also use SVM for
regression and for more than two classes. Check out the [blog post of
Usman
Malik](https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn_ for how to get started using `scikit-learn`. 

Also LDA can be generalized, which is discussed in the rest of Chapter
12, from Section 12.4 onward. FDA stands for _flexible discriminant
analysis_ and PDA for _penalized discriminant analysis_ (Figure 12.11
is particularly informative about how PDA improves upon LDA). An
example of FDA using python is available at [Jonathan Taylor's applied
statistics course
website](http://statweb.stanford.edu/~jtaylo/courses/stats306B/fda.html).

### Homework 12

Pick either (a variant of) SVM or a **generalization** of LDA and apply it
on your project data. Remember to analyze properly the effects of
parameters and design choices in the prediction error.

## Prototypes and neighbors

Why bother with models when we can just "go with the closest match"?
The idea here is to either pick from the existing labeled data a
"close-by representative" and use its label for new data that happens
to be similar to it or to _construct_ such representatives (called
_prototypes_) from the known data points.

In k-means clustering, we compute `k` _cluster centers_ by minimizing
the distance of each known data point to its nearest center. Start
with `k` random points in the feature space, for example, assign each
data point to the nearest one, and then update the center by averaging
over the members of the cluster, and repeat until some convergence
condition is met or you run out of time or some other stopping
condition tells you to stop. There are, of course, tons of variants on
how exactly to carry out this idea.

The absolute most famous person to have worked at the department where
I studied, Teuvo Kohonen, proposed a methods called LVQ (learning
vector quantification) in which the representatives are gradually
moved towards the boundaries, starting with a random assignment of
prototypes and then little by little moving the prototypes towards
randomly selected training samples at a decreasing rate. 

Another similar approach is the _Gaussian mixture method_ where the
clusters are represented as Gaussian densities and one iteratively
adjusts the weighted means and covariances (data points that are
equidistant from two or more clusters contribute to each one of them).

As mentioned before, KNN just uses the `k` closest known data points
to assign a label to a previously unseen one; we can use a majority
vote or a (weighted) mean or some other criteria of that kind.

For all of these methods, the absolute most relevant choices are
+ how distances are measured in the feature space or a transformed version thereof
+ how many clusters / neighbors are taken into account (either of
these could be somehow auto-adjusted based on prediction
error). Section 13.3.3 presents the interesting option of _tangent
distance_ to avoid some known issues.

### Homework 13

After a complete read-through of Chapter 13, make a comparison between
(some variants) of k-means, LVQ, Gaussian mixtures, and KNN for your
project data. Since these models are pretty simple, implement at least
one of them fully from scratch without using a library that already
does it for you.

## Unsupervised learning

Now we no longer have or need pre-labeled training samples. No more
`Y` in the equations, no more prediction in that same sense. We just
have the `X` now, possibly with a lot more features and a lot more
data points. The goal is to somehow **characterize** the dataset.

Some ways to go about this:

+ estimating the _probability density_ `Pr(X)` (Section 14.1)
+ _association rules_ that describe sets of feature values that tend to appear together (Section 14.2)
+ label all of it as 'true', mash it up with random data labeled as
  'false', and then apply regression (Section 14.2.4) or some other supervised method (Section 14.2.6)

A building block for these is using binary indicator variables to
whether or not a feature of an input is "close" to a specific
"typical" value for that feature. Then one searches for sets of inputs
that have the same indicators set to true (we will say that these
inputs "match" the indicator set). The _support_ of a subset of such
indicators refers to the proportion of data points that match it. One
wants to find rules that have a support larger than a threshold (for
example with the _a priori algorithm_ discussed in Section
14.2.2). The interest is in figuring out which indicator subsets are
simultaneously present with high probability (think of the peanut
butter and jelly sandwich example of the aforementioned section). The
_confidence_ of a rule `A -> B` is the support of `A -> B` normalized
by the support of `A`. An end user will typically want to manually
query (or automatically mine for) rules involving specific indicators
with both support and confidence exceeding some thresholds.

Another way to approach this issue, a long-time personal interest of
mine, is through **clustering** (Section 14.3):
+ define a pairwise similarity measure (hopefully symmetrical) 
+ use the corresponding dissimilarity as a distance (hopefully meeting the triangle inequality)
+ or [define a distance](https://link.springer.com/book/10.1007/978-3-642-30958-8) and use it's inverse of some sorts as a similarity measure
+ build a proximity matrix in these terms
+ figure out a way to define similarity and/or dissimilarity for _subsets_ of data points
+ apply a _clustering algorithm_ to group the data into (possibly
  overlapping and/or hierarchical) groups that are internally similar
  but dissimilar between groups (the belonging into a group could also
  be fuzzy or probabilistic), with something simple like k-means or
  some other approach.
  
There is a plethora of clustering methods for data points as such, let
alone graph representations based on thresholding the proximity matrix
to obtain an adjacency relation. See _spectral clustering_ in Section
14.5.3 on how to identify clusters based on the eigenvalues of the
graph Laplacian. Also the PageRank algorithm that Google uses to
determine which websites are relevant is based on an eigenvector; this
is discussed in Section 14.10 and can be applied to determine relative
importance of elements in any proximity matrix.
  
The utility of k-means for image processing is discussed in Section
14.3.9 regarding _vector quantization_.  Variants of k-means that go
beyond the usual Euclidean distance and averaging are numerous; see
for example Section 14.3.10.

Iteratively grouping the data set gives rise to a hierarchy, best
visualized as a _dendrogram_ (see Figure 14.12). These can be built
with both **top-down** approaches, dividing the data into two or more
groups in each step, or **bottom-up** by combining in each step two or
more subsets starting with singletons.

Another contribution of Teuvo Kohonen are _self-organizing maps_
(Section 14.4) that are similar to k-means but constrained into a
low-dimensional projection of a sort (a manifold referred to as a
constrained topological map). The idea is to 'bend' a 2D plane of
_principal components_ (Section 14.5) to a grid of prototypes in terms
of a sample neighborhood. 

See Sections 14.5.4 and 14.5.5 for fancy ways to apply principal
component analysis. Other ways to make use of matrix decompositions
include ICA (independent component analysis, Section 14.7) and other
ways to map stuff down into a very low dimension include
_multidimensional scaling_ (Section 14.8).

### Homework 14

After reading the whole chapter, pick any three techniques introduced
in it and apply them to your data. Make use of as many libraries as
you please in this occasion. Discuss the drawbacks and advantages of
each of the chosen techniques.

## Random forests

As discussed in Section 8.7 (bagging and bootstrap), one can improve
upon a model by actually using several models (in parallel, if you
wish) and then letting those average or vote for the outcome in some
way. This is similar to boosting (Chapter 10) that differs from the
former in that it is an iterative method that improves from one
iteration to the next. Now we look into building a bunch of
un-correlated trees and calling it a forest; the outcome is averaged
over the individual trees.

The main idea is to take a bootstrap sample as before, and then train
a tree for that sample. Let these trees carry out a majority vote when
a prediction is made.

Each individual tree is made by selecting (independently and
uniformly) at random a subset of `m` variables (the order of this
subset is a parameter), pick the best split point among those `m`
options, and then divide the data recursively until the amount of data
points in the branch falls beneath a defined threshold (also a model
parameter). The smaller the value of `m`, the less relation between
the trees is to be expected.

We can use the splitting steps to determine how important the
variables are in terms of how good the splits are (see Figure 15.5 of
the book). Also a _proximity matrix_ can be derived (a bit like a
dendrogram distance) in terms of how many shared nodes the two
variables have.

### Homework 15

After carefully reading all of Chapter 15 (regardless of how much of
Section 15.4 results comprehensible), train (and evaluate) a random
forest on your project data and compute also the variable importance
and the proximity matrix corresponding to the forest.

## Ensemble learning

There are also other ways to take a bunch of models and combine their
results. Anything that trains numerous models and uses their output to
build a prediction is in essence an ensemble method. In this chapter,
instead of just grabbing a bunch of models at random, an actual
_search procedure_ in the "learner space" is used to form that
population.

Instead of using as-short-as-possible binary codes for the classes and
then the bit positions to train two-class models, one option is to use
an error-correcting code (meaning that extra bits are added to keep
the class identifier vectors "as far apart as possible") or just
random long binary strings. 

In **forward stagewise linear regression**, assume initially that all
coefficients are zero and iteratively increment the ones that have the
highest impact (using the sign of the change they produce in the
difference between the intended label and the resulting prediction),
with a gradually slowing learning
rate. ([Tibshirani](http://www.stat.cmu.edu/~ryantibs/papers/stagewise.pdf)
discusses this very clearly and in-depth). Ideally, non-relevant
features will remain at zero coefficients whereas the relevant ones
converge towards their "true" values.

In a similar manner, in ensemble learning, one could seek to get rid
of those ensemble members that fail to contribute. A post-processing
(with potential discarding) can achieve this, as is discussed in
Section 16.3. One good option is LASSO (least absolute shrinkage and
selection operator) that does both regularization and variable
selection (check out this [tutorial by Aarsjhay
Jain](https://www.analyticsvidhya.com/blog/2016/01/ridge-lasso-regression-python-complete-tutorial/))
to get rid of unwanted ensemble members (using the ensemble members
outputs as the input variables for the prediction).

### Homework 16

Two things: examine how increasing the length of the binary string to
identify each class in a multi-class problem affects the performance
and carry out some sort of pruning on an ensemble method of your
choice, both with the project data.

## Graphs 

:green_heart::green_heart::green_heart::green_heart::green_heart::green_heart::green_heart:

So, let's put the features as vertices of a graph and also throw in
the variables we want to predict. Then, add edges using some pairwise
measurement (conditional dependencies or the like); these could also
be directed in the case that the chosen measure were
asymmetric. _

Bayesian networks_ are one way to work in this fashion. Another
approach are _Markov graphs_ where the **absence** of an edge
represents conditional independence (given the values of the other
variables). Interesting subsets of variables (vertices) are those that
_separate_ the rest of the vertex sets into two in such a way that all
paths from one to another pass through the separating subset. We can
reason about the graph in terms of its maximal _cliques_ (complete
subgraphs), assigning a potential function to each one.  

One may either already know the structure of the graph and wish to
determine some parameters, or both the structure and the parameters
need to be estimated from the data. For the latter, LASSO comes in
handy again to assume a complete graph and then rid oneself of the
edges that have zero coefficients by only working with the non-zero
ones (see Figure 17.5).

For _continuous_ variables, Gaussian models are common, whereas for
_discrete_ variables, the special case of binarization brings us to
Ising models (briefly discussed at the end of the [Algorithm
course](https://elisa.dyndns-web.com/teaching/aa/2020.html) in the
context of phase transitions). 

Also [_hidden Markov
models_](https://web.stanford.edu/~jurafsky/slp3/A.pdf) are a cool
field of study, especially for sequence-processing. A favorite of our
friend, [Arturo
Berrones](https://scholar.google.com.mx/citations?hl=en&user=RwupfCkAAAAJ),
are _Bolzmann machines_ (see Section 17.4) that are bipartite graphs
(layers of nodes with no internal edges in any of the layers, much
like a layered neural network). Section 17.4.4 shows an interesting
example of how powerful restricted Bolzmann machines can be for
unsupervised learning.

### Homework 17

Using either an existing graph-based model or one of your own
creation, build a graph of the features (possibly with
transformations, kernels or the like to expand the vertex set) and the
variables of interest for your project data. Draw this graph using
color and size to emphasize the relative importance of the variables
(vertices) and their dependencies (edges). 

## High dimensionality

When we have tons more features than we do data points, things get
challenging; this is often the case in medical studies (dozens or
maybe hundreds of patients, but possibly thousands of measurements on
each). Regularizing diagonal covariances in LDA may be useful (Section
18.2) or quadratic regularization with a linear classifier (Section
18.3). Feature selection becomes a must, really.

### Final project

Using everything you have learned about your project data during the
17 homework assignments, write an article (as you would for a
scientific journal) of your absolute best effort of applying
statistical learning to the project data. Respect the usual structure
of an article and the style of scientific writing in computational
sciences. 

Do **not** try to fit everything you did during the semester into the
article. Be smart and use what you learned in later homeworks to
improve upon the results you obtained in earlier ones instead of just
copying and pasting homework fragments together. It is recommendable
to include a comparison of techniques instead of just one technique,
but try to not exaggerate on how many techniques to include. It is
especially interesting if you manage to combine two or more techniques
into a novel adaptation for your particular situation.

Include pseudocodes (with an appropriate LaTeX package) or _very clear
and concise_ code fragments (using the `listings`package) of the
applied methods, clear equations for anything that can be expressed
mathematically, and pay extra attention to the quality of the
scientific visualization of your results. Each figure or table should
serve a clear purpose and needs to be discussed in the text; if there
is nothing of interest to conclude about it, then it should not really
be included.

Remember to properly cite the state of the art and to provide the
necessary concepts and notation in a background section. If the
feedback on the homework was helpful, you can include me in the
_acknowledgments_ section along with people who provided data or had
helpful discussions with you during the work; only people who actually
**write** (either the manuscript or code) should be listed as authors,
in all honesty (that's literally what being an author means).
