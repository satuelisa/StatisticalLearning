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
the [book's website](https://www-stat.stanford.edu/ElemStatLearn). I
use GitHub in dark mode, so the equations are set to have a black
background with white text. I hope that there will be native LaTeX
rendering in the markdown soon, as they are a bit ugly at present.

## Weekly schedule

+ [Chapter 1: Introduction](#introduction)
+ [Chapter 2: Supervised learning](#supervised-learning)
  * [Section 2.2: Least squares for linear  models](#least-squares-for-linear-models)
  * [Section 2.3: Nearest neighbors](#nearest-neighbors)

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

+ input variable (often a vector) ![](https://latex.codecogs.com/gif.latex?\bg_black&space;X) 
+ quantitative output ![](https://latex.codecogs.com/gif.latex?\bg_black&space;Y) 
+ qualitative output ![](https://latex.codecogs.com/gif.latex?\bg_black&space;G) 
+ quantitative
z		prediction ![](https://latex.codecogs.com/gif.latex?\bg_black&space;\hat{Y}) 
+ qualitative
prediction ![](https://latex.codecogs.com/gif.latex?\bg_black&space;\hat{G}) 
+ quantitative training
  data
  ![](https://latex.codecogs.com/gif.latex?\bg_black&space;(x_i,&space;y_i))
  ![](https://latex.codecogs.com/gif.latex?\bg_black&space;i&space;\in&space;1,\ldots,n) 
+ qualitative training data ![](https://latex.codecogs.com/gif.latex?\bg_black&space;(x_i,&space;g_i)) 
		 ![](https://latex.codecogs.com/gif.latex?\bg_black&space;i&space;\in&space;1,\ldots,n) 

### Least squares for linear models

The output is predicted as a weighted linear combination of the input
vector plus a constant, where the weights and the constant need to be
learned from the data. 

We can avoid dealing with the constant separately by adding a unit
input
```python
import numpy as np	
x = np.transpose(np.array([1, 2, 3, 4])) # column vector 4 x 1                  
n = len(x) # three features plus the constant                                   
w = uniform(size = n) # four weights (random for now)                           
yp = np.inner(x, w) # inner product of two rows        
print(yp)
```
and the **quality** of the prediction is compared as a sum of squares
between the desired values `y` and the predicted values `yp`. 
```python
w = np.transpose(w) # also as a column vector 4 x 1                             
yp = np.matmul(np.transpose(x), w)  # (1 x 4) x (4 x 1) = 1 x 1                 
print(yp)
``` 
Lets use matrices to make this more compact:

+ `X` is a matrix where each _row_ is an input vector and each column
  is a feature
+ `y` is a vector of the intended outputs (the first element for the
  first row of `X`, the second for the second row, etc.)
+ `yp` is then `X` multiplying the weight vector

```python
X = np.array([[1, 2, 3, 4], [1, 3, 5, 7], [1, 8, 7, 3]]) # 3 x 4                
y = np.transpose(np.array([0.9, 1.4, 1.3])) # 3 x 1, one per input    

def rss(X, y, w):
    yp = np.matmul(X, w) # predictions for all inputs                           
    return np.matmul(np.transpose(y - yp), (y - yp))
```

The best model in this sense is the one that minimizes RSS
```python
for r in range(10): # replicas                                                  
    print(rss(X, y, uniform(size = n)))	# the smaller the better     
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
 
