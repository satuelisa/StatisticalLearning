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

+ [Chapter 1: Introduction](#chapter-1)
+ [Chapter 2: Supervised learning](#chapter-2)

## Chapter 1: Introduction

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


## Chapter 2

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

to be done soon
	

