---
layout: post
title: "Assignment 2"
category: 'Homework'
---


### Question 1

Recall the toy "rabbit holes" problem covered in lecture 5 ("Poisson process" in two regions).

- Data: 
  - One patch of land of 1 acre has $x\_1 = 12$ rabbit holes.
  - A second, disjoint patch of 1 acre has $x\_2 = 32$ holes.
- Question: do the rabbits prefer one patch over the other?

The first model is given by:

\\begin{eqnarray}
\alpha &\sim \textrm{exp}(0.1) \\\\
x\_i \mid \alpha &\sim \textrm{Poi}(\alpha),
\\end{eqnarray}

and the second model is given by:

\\begin{eqnarray}
\beta\_i &\sim \textrm{exp}(0.1) \\\\
x\_i \mid \beta\_i &\sim \textrm{Poi}(\beta\_i).
\\end{eqnarray}

Analyse this problem with two different Bayesian model comparison methods. Compare the efficiency of the two approximations.


### Question 2

Consider a target distribution given by a product of $n$ normal distribution with mean zero and variance one. Consider (1) an SIS algorithm, (2) an SMC algorithm, both using a normal proposal with variance $1.2$. 

- Using the asymptotic results shown in class, derive an approximation of $\textrm{Var}(\hat Z\_n)/Z\_n^2$ for (1) and (2). 
- How many particles are required to obtain a relative variance of $0.01$?


### Question 3

Write down the general form of the reversible jump acceptance ratio for the non-conjugate DP sampler covered in lecture 12 and 13.


### Question 4 (optional)

Complete the missing parts of the code covered in the tutorial. See ``Lab 1`` under the ``Activities`` tab.