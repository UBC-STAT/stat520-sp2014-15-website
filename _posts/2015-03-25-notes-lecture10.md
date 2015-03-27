---
layout: post
title: "Lecture 10: Reversible jump MCMC"
category: 'Lecture'
---

### Motivation for Reversible jump MCMC (RJMCMC)

**Main motivation:** model choice. 

**Recall: notation**

- $I$: an index over a discrete set of models.
- $\Zscr_i$ for $i\in I$: latent space for model $i$.
- $p\_i$, $\ell\_i$, $m\_i$: prior, likelihood, and marginal likelihood densities for model $i$.

**Recall: key idea of model choice**

Put a prior $p$ on $I$, and make the uncertainty over models part of the probabilistic model.

The new joint probability density is given by:

\\begin{eqnarray}
p((i, z), x) = p(i) p\_i(z) \ell\_i(x | z),
\\end{eqnarray}

where $(i, z)$ is a member of a new latent space given by:

\\begin{eqnarray}\label{eq:new-latent-space}
\Zscr = \bigcup\_{i\in I} \left( \\{i\\} \times \Zscr\_i \right),
\\end{eqnarray}

**Recall: model saturation**

- Instead of defining the global latent space as a union of each model's latent space, define it as a product space,
- and add to that an indicator $\mu$ that selects which model to use to explain the data. The event $M\_1$ corresponds to $\mu = 1$ and $M\_2$, to $\mu = 2$. 

This creates the following auxiliary latent space:

\\begin{eqnarray}
\Zscr' = \\{1, 2\\} \times \Zscr\_1 \times \Zscr\_2.
\\end{eqnarray}

### Idea, and comparison to model saturation

- Stay in the union space, but make the dimensionality of the space in the union match.
- "Pad" with auxiliary iid random variables.

**Key advantage:** 

- We do not to instantiate all the auxiliary random variable.
- Lazy computation: only sample these auxiliary random variable when they will be needed.
- This means we can have an infinite number of auxiliary variables!
- This becomes important when $I$ is countable infinite, e.g. for non-parametric models.

### Towards RJMCMC: an alternate view to standard Metropolis-Hastings (MH).

**Recall: the MH ratio allows to transform a proposal into a Markov chain with a prescribed stationary distribution.

\\begin{eqnarray}
\frac{\pi(x')}{\pi(x)} \frac{q(x\mid x')}{q(x'\mid x)}.
\\end{eqnarray}

**Exercise:** (more error-prone than it first looks!) 

- Let $\pi(v)$ be an exponential random variable with rate 1. 
- Consider the proposal, which, given a current value of $v^{(i)}$, propose the next candidate $v^\star$ as follows:
   - Sample a multiplier $m$ with density $g(m) = 1/(\lambda m)$ on the interval $[1/e^{\lambda/2}, e^{\lambda/2}]$.
   - Return $v^\star = m \cdot v$.
- Compute the MH ratio for this proposal.

**First view:** computing $q(v^\star\mid v)$...

**Second view:**

- Auxiliary space with states of the form $x = (m, v)$.
- Two moves:
   - Sample a new value for $m \sim g(\cdot)$. (always accepted)
   - Propose one state according to a deterministic function $\psi(m, v) = (m^\star, v^\star)$, where $m^\star = 1/m$ and $v^\star = mv$. (accept-reject)
   
**Questions:** 

- What is $x^{\star\star} = \Psi(\Psi(x))$?...
- What is the acceptance ratio for the deterministic proposal?...
- Conditions for that to work?

### RJMCMC

RJMCMC works similarly to the second view of MH, with the difference that:

- We pad a variable number of auxiliary variables in order to be able to build diffeomorphic mappings $\Psi$ (more specifically, mappings with non-vanishing Jacobians).
- We many need more than one $\Psi\_j$, selected at random according to some probabilities $\rho\_{\cdot\to j}$.

**Dimensionality matching:** a necessary conditions for the mapping to be diffeomorphic is that the input dimensionality of $\Psi$ should match the output dimensionality of $\Psi$.

**Consequence:** let us say that we want to "jump" from a model with $m\_1$ dimensions into one with $m\_2$ dimensions. What constraints do we have on the number $n\_1$ of auxiliary variables we add to the first model, and the number $n\_2$ we add to the second? 

**Notation:** 

- $p(i)$ prior on model $i$
- $\pi\_i$ posterior given model $i$
- $i,i'$ old and proposed model indices
- $x, x'$ old and proposed model parameters
- $u\_i$: auxiliary variables before the move, input into $\Psi\_j$, with density $g\_i$
- $u\_{i'}$: auxiliary variables after the move, output of $\Psi\_j$, with density $g\_{i'}$

**Ratio for RJMCMC:**

\\begin{eqnarray}
\frac{p(i')\pi\_{i'}(x')}{p(i)\pi\_i(x)} \frac{\rho\_{i'\to i}}{\rho\_{i\to i'}} \frac{g\_{i'}(u\_{i'})}{g\_{i}(u\_{i})} \left| J(x', u\_2) \right|
\\end{eqnarray}

**Example:** textbook, page 365.

