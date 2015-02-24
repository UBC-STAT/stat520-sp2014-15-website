---
layout: post
title: "Assignment 1"
category: 'Homework'
---


**Note:** in construction! You can start looking and thinking about the questions, but note that additions/corrections may appear this week. 

Question 1
==========

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


Food for thoughts (not in the assignment): 

- What if there is a very large number of versions (e.g. all configurations of background colour, text colour, fonts, etc)?