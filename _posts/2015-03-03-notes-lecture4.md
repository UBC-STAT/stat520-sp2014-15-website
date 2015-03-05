---
layout: post
title: "Lecture 4: Mixtures and hierarchies"
category: 'Lecture'
---


### Prior choice and hierarchical models

We have touched on a first method to deal with prior hyper-parameters in this exercise. Here we discuss alternative methods.

Since conjugacy leads us to consider families of priors indexed by a hyper-parameter $h$, this begs the question of how to pick $h$. Note that both $m\_h(x)$ and $p\_h(z | x)$ implicitly depend on $h$. Here are some guidelines for approaching this question:

1. One can maximize $m\_h(x)$ over $h$, an approach called **empirical Bayes**. Note however that this does not fit the Bayesian framework (despite its name).
2. If the dataset is moderate to large, one can test a range of reasonable values for $h$ (a range obtained for example from discussion with a domain expert); if the action selected by the Bayes estimator is not affected (or not affected too much), one can side-step this issue and pick arbitrarily from the set of reasonable values.
3. If it is an option, one can collect more data *from the same population*. Under regularity conditions, the effect of the prior can be decreased arbitrarily (this follows from the celebrated Bernstein-von Mises theorem, see van der Vaart, p.140).
4. If we only have access to other datasets that are related (in the sense that they have the same type of latent space $\Zscr$), but potentially from different populations, we can still exploit them using a **hierarchical Bayesian** model, described next.

Hierarchical Bayesian models are conceptually simple: 

1. We create distinct, exchangeable latent variables $Z\_j$, one for each related dataset $X\_j$
2. We make the hyper-parameter $h$ be the realization of a random variable $H$. This allows the dataset to originate from different populations.
3. We force all $Z\_j$ to share this one hyper-parameter. This step is crucial as it allows information to flow between the datasets.

<img src="{{ site.url }}/images/hierarchical-lec3-fixedcap.png" alt="Drawing" style="width: 400px; float: center"/>

One still has to pick a new prior $p^*$ on $H$, and to go again through steps 1-4 above, but this time with more data incorporated. Note that this process can be iterated as long as there is some form of known hierarchical structure in the data (as a concrete example of a type of dataset that has this property, see this non-parametric Bayesian approach to $n$-gram modelling: [Teh, 2006](http://acl.ldc.upenn.edu/P/P06/P06-1124.pdf)). More complicated techniques are needed when the hierarchical structure is unknown (we will talk about two of these techniques later in this course, namely hierarchical clustering and phylogenetics).

The cost of taking the hierarchical Bayes route is that it generally requires resorting to Monte Carlo approximation, even if the initial model is conjugate.

### Mixture models. 

**Example:** Modelling the weight of walruses. 

Observation: weights of specimens $x\_1, x\_2, ..., x\_n$. 

Inferential question: what is the most atypical weight among the samples?

**Method 1:** 

- Find a normal density $\phi\_{\mu, \sigma^2}$ that best fits the data. 
- Bayesian way: treat the unknown quantity $\phi\_{\mu, \sigma^2}$ as random.
- Equivalently: treat the parameters $\theta = (\mu, \sigma^2)$ as random, $(\sigma^2 > 0)$. Let us call the prior on this, $p(\theta)$ (for example, another normal times a gamma, but this will not be important in this first discussion).

Limitation: fails to model sexual dimorphism (e.g. male walruses are typically heavier than female walruses). 

Solution: <img src="{{ site.url }}/images/walrus-plot-l2.png" alt="Drawing" style="width: 200px; float: right"/>

**Method 2:** Use a *mixture model*, with two mixture components, where each one assumed to be normal.

- In terms of density, this means our modelled density has the form:
\\begin{eqnarray}
\phi = \pi \phi\_{\theta\_1} + (1 - \pi) \phi\_{\theta\_2},
\\end{eqnarray}
for $\pi \in [0, 1]$.
- Equivalently, we can write it in terms of auxiliary random variables $Z\_i$, one of each  associated to each measurement $X\_i$: 
\\begin{eqnarray}
Z\_i & \sim & \Cat(\pi) \\\\
X\_i | Z\_i, \theta\_1, \theta\_2 & \sim & \Norm(\theta\_{Z\_i}).
\\end{eqnarray}
- Each $Z\_i$ can be interpreted as the sex of animal $i$.

Unfortunately, we did not record the male/female information when we collected the data!

- Expensive fix: Do the survey again, collecting the male/female information
- Cheaper fix: Let the model guess, for each data point, from which cluster (group, mixture component) it comes from. <img src="{{ site.url }}/images/directed-graph-l2.png" alt="Drawing" style="width: 200px; float: right"/>

Since the $Z\_i$ are unknown, we need to model a new parameter $\pi$. Equivalently, two numbers $\pi\_1, \pi\_2$ with the constraint that they should be nonnegative and sum to one. We can interpret these parameters as the population frequency of each sex. We need to introduce a prior on these unknown quantities.

- Simplest choice: $\pi \sim \Uni(0,1)$. But this fails to model our prior belief that male and female frequencies should be close to $50:50$.
- To encourage this, pick a prior density proportional to: <img src="{{ site.url }}/images/beta.jpg" alt="Drawing" style="width: 200px; float: right"/>
\\begin{eqnarray}
p(\pi\_1, \pi\_2) \propto \pi\_1^{\alpha\_1 - 1} \pi\_2^{\alpha\_2 - 1},
\\end{eqnarray}  
where $\alpha\_1 > 0, \alpha\_2 > 0$ are fixed numbers (called hyper-parameters). The $-1$'s give us a simpler restrictions on the hyper-parameters required to ensure finite normalization of the above expression). 
- The hyper-parameters are sometimes denoted $\alpha = \alpha\_1$ and $\beta = \alpha\_2$. 
  - To encourage values of $\pi$ close to $1/2$, pick $\alpha = \beta = 2$. 
  - To encourage this even more strongly, pick $\alpha = \beta = 20$. (and vice versa, one can take value close to zero to encourage realizations with one point mass larger than the other.)
  - To encourage a ratio $r$ different than $1/2$, make $\alpha$ and $\beta$ grow at different rates, with $\alpha/(\alpha+\beta) = r$. But in order to be able to build a bridge between Dirichlet distribution and Dirichlet processes, we will ask that $\alpha = \beta$ 

**Dirichlet distribution:** This generalizes to more than two mixture components easily. If there are $K$ components, the density of the Dirichlet distribution is proportional to:
\\begin{eqnarray}
p(\pi\_1, \dots, \pi\_K) \propto \prod\_{k=1}^{K} \pi\_k^{\alpha\_k - 1},
\\end{eqnarray} 
Note that the normalization of the Dirichlet distribution is analytically tractable using Gamma functions $\Gamma(\cdot)$ (a generalization of $n!$). See the wikipedia [entry](http://en.wikipedia.org/wiki/Dirichlet_distribution) for details.



<!-- ### Second example: aspirin clinical trial

- http://jackman.stanford.edu/classes/350C/07/randomeffects.pdf

OR: 
http://www.hockey-reference.com/leagues/NHL_2014_goalies.html

http://www.stat.cmu.edu/~acthomas/724/Efron-Morris.pdf

- trick to initialize deterministic models -->
