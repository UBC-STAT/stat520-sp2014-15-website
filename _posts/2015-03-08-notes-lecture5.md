---
layout: post
title: "Lecture 5: Model choice"
category: 'Lecture'
---

### Examples

- Choice in the likelihood: should we replace the logit transform in the Challenger example by a probit transformation?
- Choice in the prior: in the challenger example, should we replace the normal distribution by a t distribution?
- Variable (covariate) selection: should we use only the temperature covariate, or both  temperature and humidity, or only humidity, to predict o-ring failure?
- Determining the number of clusters in mixture models. 
  - Example: cancer heterogeneity and bulk sampling. 
  - Issue: observed vs. population cluster count.

### Notation

- $I$: an index over a discrete set of models.
- $\Zscr_i$ for $i\in I$: latent space for model $i$.
- $p\_i$, $\ell\_i$, $m\_i$: prior, likelihood, and marginal likelihood densities for model $i$.

### Key idea

Put a prior $p$ on $I$, and make the uncertainty over models part of the probabilistic model.

The new joint probability density is given by:

\\begin{eqnarray}
p((i, z), x) = p(i) p\_i(z) \ell\_i(x | z),
\\end{eqnarray}

where $(i, z)$ is a member of a new latent space given by:

\\begin{eqnarray}\label{eq:new-latent-space}
\Zscr = \bigcup\_{i\in I} \left( \\{i\\} \times \Zscr\_i \right),
\\end{eqnarray}



**Notation:** denote the event that model $i$ is the model explaining the data by $M\_i$.

**Outcome:** Using this construction, model choice can in principle be approached using the same methods as those used last week.

**Example:** (TODO)

- Model selection with 0-1 loss.
- Model averaging for prediction.

### Observations

**Graphical modelling:** Equation (\ref{eq:new-latent-space}) cannot be directly expressed as a non-trivial graphical model (since it is not a product space). How to transform it into a graphical model?

**Non-regularity:** even with the reductions introduced so far, model selection justifies special attentions because of non-regularities: the likelihood depends in a non-smooth way upon the model indicator variable. Importantly, different models have different dimensionality of the latent space. We will see that MCMC then requires special techniques called trans-dimensional MCMC. 

### Bayes factor

Ratio of the marginal likelihood for two models:

\\begin{eqnarray}\label{eq:bayes-factor}
B\_{12} = \frac{m\_1(x)}{m\_2(x)}
\\end{eqnarray}

Values of $B\_{12}$ greater than 1.0 favor model #1 over #2. Values smaller than 1.0 favor #2 over #1.

This is just a reparameterization of the Bayes estimator with an asymmetric 0-1 loss. Note that it is different from a likelihood ratio:

\\begin{eqnarray}\label{eq:likelihood-ratio}
\frac{\sup\_{z\_1} \ell\_1(x|z\_1)}{\sup\_{z\_2} \ell\_2(x|z\_2)},
\\end{eqnarray}

which does not arise within the Bayesian framework.

### Computation of marginal likelihoods and Bayes factors


#### Conjugate models

Recall our notation:

- $h$ is a hyper-parameter for the prior, $p\_h(z)$.
- Conjugacy means that the posterior density coincides with the prior for updated hyperpameters $u(x, h)$: $p\_h(z|x) = p\_{u(x, h)}(z)$.

Rearranging Bayes rule:

\\begin{eqnarray}
m(x) & = & \frac{p\_{h}(z) \ell(x | z)}{p(z | x)} \\\\
& = & \frac{p\_{h}(z) \ell(x | z)}{p\_{u(x, h)}(z)}.
\\end{eqnarray}

Since this is true for all $z$, we can pick an arbitrary $z\_0$, and evaluate each component of the right-hand side by assumption.

**Example:** Poisson process on two regions.

**Pro:** exact.

**Con:** only possible for tractable conjugate families.

#### Model saturation

The idea is to build an augmented model, which can be written as a graphical model (in contrast to Equation (\ref{eq:new-latent-space}), and from which we can still approximate $m\_i(x)$.

**Construction of the auxiliary latent space:**

- Instead of defining the global latent space as a union of each model's latent space, define it as a product space,
- and add to that an indicator $\mu$ that selects which model to use to explain the data. The event $M\_1$ corresponds to $\mu = 1$ and $M\_2$, to $\mu = 2$. 

This creates the following auxiliary latent space:

\\begin{eqnarray}
\Zscr' = \\{1, 2\\} \times \Zscr\_1 \times \Zscr\_2.
\\end{eqnarray}

**Example:** dim($\Zscr\_1$) = 1, dim($\Zscr\_2$) = 2. What is a picture for $\Zscr'$? Contrast with $\Zscr$ from Equation (\ref{eq:new-latent-space}).

**Construction of the auxiliary joint distribution:** suppose the current state is $(\mu, z\_1, z\_2, x)$. We need to define an auxiliary joint density $\tilde p(\mu, z\_1, z\_2, y)$. 

The idea is that when $\mu = 1$, we explain the data $x$ using $z\_1$, and when $\mu = 2$, we explain the data $x$ using $z\_2$. 

In notation, if $\mu = 1$, we set:

\\begin{eqnarray}
\tilde p(\mu, z\_1, z\_2, x) = p(\mu) p\_1(z\_1) p\_2(z\_2) \ell\_1(x | z\_1),
\\end{eqnarray}

and if $\mu = 2$,

\\begin{eqnarray}
\tilde p(\mu, z\_1, z\_2, x) = p(\mu) p\_1(z\_1) p\_2(z\_2) \ell\_2(x | z\_2).
\\end{eqnarray}

**Exercise:** show that the marginal $\tilde p(\mu | x)$ can be used to obtain the Bayes factor:

\\begin{eqnarray}
\frac{\tilde p(1 | x)}{\tilde p(2 | x)} \frac{p(2)}{p(1)} = B_{12}
\\end{eqnarray}

 
**Pro:** can use existing MCMC methods such as JAGS/Stan without changing the MCMC code.

**Cons:** 

- Limited to finite collections of models, $|I| < \infty$.
- Can be slow.  


