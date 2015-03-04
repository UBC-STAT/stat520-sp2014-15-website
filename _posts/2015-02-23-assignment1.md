---
layout: post
title: "Assignment 1"
category: 'Homework'
---

Assignment is due on Wednesday, March 11.

### Question 1

In its early days, Google used a method in the spirit of what is described below to tweak their website. Let us say that they want to decide if they should use a white background or a black background (or some other cosmetic feature). 

- Each time enters the website for the first time, flip a fair (virtual) coin.
- If it is head, show the website with a black background (option A), else, show the one with a white background (option B).
- Record if the user click on at least one search result (as a surrogate to whether they "liked" the website or not). 

After a short interval of time, they record the following results:

- number of times A was shown: 51
- number of times A was shown and the user clicked: 45
- number of times B was shown: 47
- number of times A was shown and the user clicked: 37

Using this toy data, answer the following questions:

1. Derive a probabilistic model suitable for a Bayesian analysis of this problem.
2. Derive the Bayes estimator for deciding which of the two options to use.
3. Use a probabilistic language to make a decision from the data above.

Food for thoughts (not in the assignment): what if there is a very large number of versions (e.g. all configurations of background colour, text colour, fonts, etc)?

Note: this question can be solved with either JAGS or Stan.

### Question 2

Start by reading the section *context* in the notes of lecture 3. Report:

1. Posterior densities on the parameters.
2. An answer to the question investigated (again, see section *context* in the notes of lecture 3).
3. Compare and interpret the uncertainty of the answers for 1 and 2 above.
4. An important knob to tune when designing this model is the hyperparameters for the priors on ``slope`` and ``intercept`` parameters. A simple strategy to do this tuning is to do the inference with different values and see if the conclusion is stable for a range of extreme values. Do this by varying the order of magnitude of the variance hyperparameters (i.e. trying $1, 0.1, 0.01, \dots$ while keeping the value of the mean hyperparameter set to zero).

Note: this question can be solved with either JAGS or Stan.

### Question 3

Look at the data in the file ``texting-data.csv``. The $n$-th row represents the number of texts sent by a certain cellphone user on day $n$. Use a mixture model and JAGS to find a point in time where the user's texting habits changed in that period. Report the posterior over the point in time where the habits changed. To help inference, you can assume that the change point is away from the end-points of the time series.

Note: use JAGS for this problem.

### Question 4

Suppose that your goal is to predict an unobserved parameter $p$ of a Bernoulli random variable (for example, ``p_soyuz`` in the example covered in lecture 4). To make things interesting, suppose I give a prize to the best **pessimistic** prediction (respectively, optimistic), i.e. the one that is closest to the truth but not above it (respectively, not below it). Derive a Bayesian method to solve this problem. Optionally, implement your method on the launcher example dataset to compute an actual prediction.

### Theoretical questions

Summarize your solution for the following exercises covered in class:

- Prove that if $f$ and $g$ are densities, and $f$ is proportional to $g$, then $f = g$.
- Suppose $X|Z$ is normally distributed with mean $Z$ and variance one, and we put a normal prior on $Z$. Find the Bayes estimator for the KL intrinsic loss.
- (Optional) Assume now that $X|Z$ is exponential with rate $Z$, and that the posterior was approximated by some MC samples $Z^{(1)}, \dots, Z^{(N)}$. How would you approach the problem of approximating the Bayes estimator for the Hellinger intrinsic loss in this case? Hint: the Hellinger distance is given by $1 - 2\sqrt{x'x}/(x' + x)$ in the exponential case.
- (Optional) More generally, how to approximate Bayes estimators from MC samples for a "black box" loss (i.e. a loss where all you can do is do pointwise evaluation? Can this be done computationally efficiently (in the sense that the computational cost is not that much than running the MCMC chain)?
