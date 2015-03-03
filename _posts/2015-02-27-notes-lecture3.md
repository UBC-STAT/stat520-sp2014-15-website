---
layout: post
title: "Lecture 3: Hands-on Bayesian analysis"
category: 'Lecture'
---





### Probabilistic programming basics

**What probabilistic programming does:** it creates the Monte Carlo estimates we talked about last week, from a *declarative* definition of a probabilistic model.

What is a declarative language (compared to the type of languages you are used to (python, R, Java, etc); which are called *imperative*)? 

- Declarative: you program by describing the *goal* of the program.
- Imperative: you program by describing an excruciatingly detailed recipe for achieving your goal.

**In this lecture:** I will cover JAGS, and for those interested, give some pointers for learning Stan as well.

**Comparison of JAGS vs Stan:** From a practical point of view, the key difference is that Stan is much faster, but on the other hand it does not support latent discrete random variables.

Under the hood, this is because Stan uses a sampling method specialized to $\RR^n$ (Hamiltonian Monte Carlo).

**Other probabilistic programming** languages/packages:

- winBUGS: historically very important, but development now seems stalled (and not open source), so I would recommend using JAGS which has a very similar syntax.
- pyMC: very promising. Can define new function/distribution more easily than with JAGS. Syntax less intuitive.
- blang!: our own. Can define new datatypes. In construction. Let us know if you are interested to learn more about it.
- [Many more](http://probabilistic-programming.org/wiki/Home)

### First example: challenger disaster

**Context:**

- Challenger space shuttle exploded 73 seconds after launch in 1986
- Cause: O-ring used in the solid rocket booster failed due to low temperature at the time of launch (31 F [-1 C]; all temperatures in Fahrenheit from now on) 
- Question investigated: was it unsafe to authorize the launch given the temperature in the morning of the launch?

**Data:**

```
Date,Temperature,Damage Incident
04/12/1981,66,0
11/12/1981,70,1
3/22/82,69,0
6/27/82,80,NA
01/11/1982,68,0
04/04/1983,67,0
6/18/83,72,0
8/30/83,73,0
11/28/83,70,0
02/03/1984,57,1
04/06/1984,63,1
8/30/84,70,1
10/05/1984,78,0
11/08/1984,67,0
1/24/85,53,1
04/12/1985,67,0
4/29/85,75,0
6/17/85,70,0
7/29/85,81,0
8/27/85,76,0
10/03/1985,79,0
10/30/85,75,1
11/26/85,76,0
01/12/1986,58,1
```

**Things to report:**

- A simple probabilistic model to solve this problem.
- A JAGS or Stan implementation of this model.
- Posterior densities on the parameters.
- An answer to the question investigated (see section *context* above).

**Model:** [logistic regression](http://en.wikipedia.org/wiki/Logistic_regression). Input: temperature. Output: failure indicator variable.

**Main idea in JAGS (and Stan):** Statements define a random variable (say $X$) by specifying the conditional distribution of $X$, which depends on zero, one or more other random variables.

For example, ``dunif(a, b)`` denotes the uniform distribution between $a$ and $b$. Therefore, ``X ~ dunif(Y, Z)`` means that the conditional distribution of $X$ given $Y$ and $Z$ is a uniform distribution between $Y$ and $Z$ (note: distribution names are  different in Stan, see the [Stan manual](https://github.com/stan-dev/stan/releases/download/v2.6.0/stan-reference-2.6.0.pdf).

An important special case arises when a random variables is a deterministic function of other random variables. For example, ``X <- Y^2 + Z`` means that the conditional distribution of $X$ given $Y$ and $Z$ is a Dirac delta on $Y^2 + Z$. In other words, $X(\omega) = (Y(\omega))^2 + Z(\omega)$.

**Exercise:** Let us first build a logistic regression model on a single observation.

1. Define a random variable called ``incident``, which, given a random variable ``p``, is Bernoulli with success probability $p$.
2. Set $p$ as a deterministic function of two random variables, ``slope`` and ``intercept``, to complete the specification of a logistic regression model.
3. Set normal priors on the variables ``slope`` and ``intercept``. See the [JAGS manual](http://www.stats.ox.ac.uk/~nicholls/MScMCMC14/jags_user_manual.pdf) and the [Stan manual](https://github.com/stan-dev/stan/releases/download/v2.6.0/stan-reference-2.6.0.pdf) to find the detailed syntax for the distributions available in each language. See also this [quick reference page for JAGS](http://www.stat.ubc.ca/~bouchard/courses/stat520-sp2014-15/tool/2014/10/06/jags.html).

**Vectors and loops in JAGS (and Stan):** 

- You can access the i-th entry of a vector name ``my_vector`` using ``my_vector[i]``. 
- To loop over indices 1 up to N, use ``for (i in 1:N) { ... }``.
- To get the dimension of a vector use ``length(my_vector)`` (in Stan, use ``rows(my_vector)`` (vectors are column vector by default)).

**Exercise:** modify your code to support several observations.

**Type declarations (Stan only):** Stan requires defining the datatypes of the parameters and of the data. In this example, this is done using:

```
data {  
  int N;
  real temperature[N]; 
  int incident[N]; 
}
parameters {
  real intercept; 
  real slope;
}
```

**Running the code (JAGS):** we will call JAGS (and Stan, see below) via R. Here is a walk through of the ``run.r`` file:

```r
require(rjags)
require(coda.discrete.utils)
require(ggplot2)
require(coda)
require(ggmcmc)

data <- read.csv( file = "challenger_data-train.csv", header = TRUE)

# Make the simulations reproducible
# 1 is arbitrary, but the rest of the simulation is 
# deterministic given that initial seed.
my.seed <- 1 
set.seed(my.seed) 
inits <- list(.RNG.name = "base::Mersenne-Twister",
              .RNG.seed = my.seed)

model <- jags.model(
  'model.bugs', 
  data = list( # Pass on the data:
    'incident' = data$Damage.Incident, 
    'temperature' = data$Temperature), 
  inits=inits) 

samples <- 
  jags.samples(model,
               c('slope', 'intercept'), # These are the variables we want to monitor (plot, etc)
               200000) # number of MCMC iterations

print(coda.expectation(samples))
coda.density(samples)
```

Some things to pay particular attention to:

- Importance of fixing the random seed. 
- What data we provide determines the probabilistic calculation. For example, here we are approximating the distribution of ``slope``, ``intercept`` given the random vectors ``incident`` and ``temperature`` (because we provide as input only these last two).
- Setting the number of MCMC iterations in ``jags.samples(...)``. Simple strategy: start at 1000 MCMC iterations, and by a factor of 10 until the conclusions do not change.
- Setting the random variables to monitor via ``c('slope', 'intercept')``.

**Running the code (Stan):**

```
require(rjags)
require(coda.discrete.utils)
require(ggplot2)
require(coda)
require(ggmcmc)
require(rstan)

data <- read.csv( file = "challenger_data-train.csv", header = TRUE)


data <- list(N = length(data$temperature), 
             temperature = data$temperature,
             incident = data$incident)

fit <- stan(file = 'model.stan', data = data, 
            iter = 100000, chains = 1)

tidy <- ggs(fit)
ggmcmc(tidy)
```

**Exercise:** complete the Challenger question in the assignment (see ``Activities`` tab on the course webpage). Also have a look to the next section for more information on visualization of MCMC samples.

### Other tools for analyzing the JAGS output in R

A few other things available from the output of ``samples <- jags.samples(...)`` via the package [coda.discrete.utils](https://github.com/alexandrebouchard/coda-discrete-utils):

- ``coda.cdf(samples)`` to plot cumulative distribution functions
- ``coda.pmf(samples)`` to plot probability mass functions (for discrete problems)
- ``coda.variance(samples)``
- ``coda.density2d(samples, "slope", "intercept")`` to inspect bivariate posterior density estimates.

If you replace ``samples <- jags.samples(...)`` by ``samples <- coda.samples(...)`` you can use the tools provided by the venerable [coda R package](http://cran.r-project.org/web/packages/coda/coda.pdf):

```
pdf("coda-plots.pdf")
plot(samples)
dev.off()
print(HPDinterval(samples))
```

Another useful package is [ggmcmc](http://xavier-fim.net/packages/ggmcmc/), which in particular allows you to transform the output into a [tidy](http://vita.had.co.nz/papers/tidy-data.pdf) data frame, and to use ggplot2 for example:

```
tidy <- ggs(samples)
ggmcmc(tidy)
```


### Prior choice and hierarchical models

We have touched on a first method to deal with prior hyper-parameters in question 2 of the assignment. Here we discuss alternative methods.

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
  - To encourage a ratio $r$ different that $1/2$, make $\alpha$ and $\beta$ grow at different rates, with $\alpha/(\alpha+\beta) = r$. But in order to be able to build a bridge between Dirichlet distribution and Dirichlet processes, we will ask that $\alpha = \beta$ 

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