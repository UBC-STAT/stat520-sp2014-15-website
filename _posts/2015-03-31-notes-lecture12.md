---
layout: post
title: "Lecture 12: DP, continued; non-conjugate inference"
category: 'Lecture'
---

#### Sampling from a sampled measure

Since the realization of a DP is a probability distribution, we can sample from this realization! Let us call these samples from the DP sample $\underline{\theta\_1}, \underline{\theta\_2}, \dots$

\\begin{eqnarray}
G & \sim & \DP \\\\
\underline{\theta\_i} | G & \sim & G
\\end{eqnarray}

**Preliminary observation:** If I sample twice from an atomic distribution, there is a positive probability that I get two identical copies of the same point (an atomic measure $\mu$ is one that assign a positive mass to a point, i.e. there is an $x$ such that $\mu(\\{x\\}) > 0$). This phenomenon does not occur with non-atomic distribution (with probability one).

The realizations from the random variables $\underline{\theta\_i}$ live in the same space $\Omega$ as those from $\theta\_i$, but $\underline{\theta}$ and $\theta$ have important differences:

- The list $\underline{\theta\_1}, \underline{\theta\_2}, \dots$ will have duplicates with probability one (why?), while the list $\theta\_1, \theta\_2, \dots$ generally does not contain duplicates (as long as $G\_0$ is non-atomic, **which we will assume today**). 
- Each value taken by a $\underline{\theta\_i}$ corresponds to a value taken by a $\theta$, but not vice-versa. 

To differentiate the two, I will use the following terminology:

- *Type:* the variables $\theta\_i$
- *Token:* the variables $\underline{\theta\_i}$

**Question:** How can we simulate four tokens, $\underline{\theta\_1}, \dots,  \underline{\theta\_4}$?

1. **Hard way:** use the stick breaking representation to simulate all the types and their probabilities. This gives us a realization of the random probability measure $G$. Then, sample four times from $G$. <img src="{{ site.url }}/images/dpSimulation-l2.png" alt="Drawing" style="width: 500px"/>
2. **Easier way:** sample $\underline{\theta\_1}, \dots, \underline{\theta\_4}$ from their marginal distribution directly. This is done via the *Chinese Restaurant Process* (CRP).

Let us look at this second method. Again, we will focus on the algorithmic picture, and cover the theory after you have implemented these algorithms.

The first idea is to break the marginal $(\underline{\theta\_1}, \dots, \underline{\theta\_4})$ into sequential decisions. This is done using the chain rule: 

1. Sample $\underline{\theta\_1}$, then, 
2. Sample $\underline{\theta\_2}|\underline{\theta\_1}$, 
3. etc., 
4. Until $\underline{\theta\_4}|(\underline{\theta\_1}, \underline{\theta\_2}, \underline{\theta\_3})$. <img src="{{ site.url }}/images/marginalization.jpg" alt="Drawing" style="width: 100px; float: right"/>

The first step is easy: just sample from $G\_0$! For the other steps, we will need to keep track of *token multiplicities*. We organize the tokens $\underline{\theta\_i}$ into groups, where two tokens are in the same group if and only if they picked the same type. Each group is called a *table*, the points in this group are called *customers*, and the value shared by each table is its *dish*.

Once we have these data structures, the conditional steps 2-4 can be sampled easily using the following decision diagram: 

<img src="{{ site.url }}/images/crp-decisions.jpg" alt="Drawing" style="width: 300px"/>

Following our example, say that the first 3 customers have been seated at 2 tables, with customers 1 and 3 at one table, and customer 2 at another. When customer 4 enters, the probability that the new customer joins one of the existing tables or creates an empty table can be visualized with the following diagram:

<img src="{{ site.url }}/images/tables-l2.png" alt="Drawing" style="width: 500px"/>

Formally:

\\begin{eqnarray}\label{eq:crp}
\P(\underline{\theta\_{n+1}} \in A | \underline{\theta\_{1}}, \dots, \underline{\theta\_{n}}) = \frac{\alpha\_0}{\alpha\_0 + n} G\_0(A) + \frac{1}{\alpha\_0 + n} \sum\_{i = 1}^n \1(\underline{\theta\_{i}} \in A).
\\end{eqnarray}

**Clustering/partition view:** This generative story suggests another way of simulating the tokens:

1. Generate a partition of the data points (each block is a cluster)
2. Once this is done, sample one dish for each table independently from $G\_0$.

By a slight abuse of notation, we will also call and denote the distribution on the partition the CRP. It will be clear from the context if the input of the $\CRP$ is a partition or a product space $\Omega \times \dots \times \Omega$.

**Important exercise: Exchangeability.** We claim that for any permutation $\sigma : \\{1, \dots, n\\} \to \\{1, \dots, n\\}$, we have:

\\begin{eqnarray}
(\underline{\theta\_1}, \dots, \underline{\theta\_n}) \deq (\underline{\theta\_{\sigma(1)}}, \dots, \underline{\theta\_{\sigma(n)}})
\\end{eqnarray}

You can convince yourself with the following small example, where we compute a joint probability of observing the partition $\\{\\{1,2,3\\},\\{4,5\\}\\}$ with two different orders. First, with the order $1 \to 2 \to 3 \to 4 \to 5$, and $\alpha\_0 = 1$:

<img src="{{ site.url }}/images/CRP-order0.jpg" alt="Drawing" style="width: 450px"/>

Then, with the order $4 \to 5 \to 3 \to 2 \to 1$, we get:

<img src="{{ site.url }}/images/CRP-order1.jpg" alt="Drawing" style="width: 450px"/>

The product of these CRPs is called [Ewens's sampling formula](http://en.wikipedia.org/wiki/Ewens%27s_sampling_formula).

#### Dirichlet Process Mixture model

- In many applications, the observations in one cluster component are not all identical!
- This motivates the Dirichlet Process Mixture (DPM) model

A DPM is specified by:

- A likelihood model with density $\ell(x|\theta)$ over each individual observation (a weight). For example, a normal distribution (a bit broken since weights are positive, but should suffice for the purpose of exposition).
- A conjugate base measure, $G\_0$, with density $p(\theta)$. As before, $\theta$ is an ordered pair containing a real number (modelling a sub-population mean) and a positive real number (modelling a sub-population variance (or equivalently, precision, the inverse of the variance)). A normal-inverse-gamma distribution is an example of such a prior.
- Some hyper-parameters for this parametric prior, as well as a hyper-parameter $\alpha\_0$ for the Dirichlet prior.

To simulate a dataset, use the following steps:

1. Break a stick $\pi$ according to the algorithm covered in the previous section.
2. Simulate an infinite sequence $(\theta\_1, \theta\_2, \dots)$ of iid normal-inverse-gamma random variables. The first one corresponds to the first stick segment, the second, to the second stick segment, etc.
3. For each datapoint $i$:
     1. Throw a dart on the stick, use this random stick index $Z\_i$. Grab the corresponding parameter $\theta\_{Z\_i}$.
     2. Simulate a new datapoint $X\_i$ according to $\ell(\cdot | \theta\_{Z\_i})$.
     
This algorithmic description has the following graphical model representation (for details on the plate notation see the [wiki page](https://en.wikipedia.org/wiki/Plate_notation)).  
<img src="{{ site.url }}/images/dpm-gm.jpg" alt="DPMGM" style="width: 300px;"/>

Equivalently:

1. Sample $(\underline{\theta\_1}, \dots, \underline{\theta\_n})$ from the CRP (where $n$ is the number of data points).
2. Simulate $X\_i$ according to $\ell(\cdot | \underline{\theta\_i})$.

#### Posterior simulation with MCMC

**Important concept:** Rao-Blackwellization.

- Reduce state space in which we do MCMC.
- Synonyms: collapsing, marginalization.
- Opposite of auxiliary variables, which enlarge (augment) the space.
- Rough comparison (not always true, i.e. the classical Rao-Blackwell does not always apply): 
  - Rao-Blackwellization typically makes each MCMC iteration more expensive, but reduce the number of MCMC iterations required to get a certain accuracy of the MC averages
  - Auxiliary variables typically go the other way around.
  - Both can be combined (enlarging some aspects and collapsing others).
  
Formally: 

- Let $s : \Xscr \to \Yscr$ be a map that **s**implifies the original state space $\Xscr$ of the MCMC into a reduced state space $\Yscr$. 
- From the original target distribution $\pi$, build a new target distribution $\pi\_\star$ defined as follows: for any $A \subset \Yscr$, let $\pi\_\star(A) = \pi(s^{-1}(A))$. When we have densities, this gives us a density $\pi\_\star(y) = \int\_{x:s(x) = y} \pi(x) \ud x$ (assuming $s$ is non pathological). 
- Now, design a proposal $q$ on the simplified space $\Yscr$ instead of $\Xscr$.
- Challenge: the MH ratio will involve $\pi\_\star$, and therefore the integral needs to be solved analytically (other options??)

Examples:

- $x = (x\_1, x\_2)$, and $s\_1(x) = x\_1$.
- $v = (\pi\_{1:\infty}, z\_{1:n})$ and $s\_2(v) = $ partition $\rho(v)$ of the samples $\underline{\theta\_{1:n}}$ from the DP. 
- $w = (\rho, \theta\_{1:\infty})$ and $s\_3(w) = \rho$.

**Rao-Blackwellization in DPMs:** with Dirichlet process priors (and Pitman-Yor priors):

1. You can always compute $\pi\_\star(\rho) = \pi(s\_2^{-1}(\rho))$ using Ewen's sampling formula (see the exercise above). 
- Moreover, **if** the parametric family $\ell$ is conjugate with $G\_0$, we can also compute $\pi\_{\star\star}(\rho) = \pi(s\_3^{-1}(\rho))$, using formula (20) from lecture 2. This is called the collapsed sampler, as all the continuous variables have been marginalized.

**Today:** let us assume only (1) above, i.e. we have a likelihood $\ell$ which may not have a tractable conjugacy relationship with $G\_0$.

**State space:** 

- $\rho$ (i.e. we marginalize/simplify $\pi\_{1:\infty}, z\_{1:n}$ into $\rho$
- One parameter $\theta$ for each table, denoted $\theta\_1, \dots, \theta_K$ (block in $\rho$). Note that $K$ is random, so the state space is a union of spaces of different dimensionalities.

**MCMC moves:**

1. Keep $\rho$ fixed, propose a change to one or several $\theta\_k$, accept-reject using the standard MH ratio.
2. Take one customer out of the restaurant. Let $K\_-$ be the number of non-empty tables after removing this customer ($K\_-$ could be either the same as $K$, or $K-1$ if the customer was sitting by themselves). We will keep the parameters of these tables fixed. Propose a new table, out of: either an existing non-empty table, or a new table. Overall, the number of table $K'$ after the move can be either $K-1$, $K$, or $K+1$.

**Note:** move (2) can change the dimensionality of the space. How to approach this can be viewed through the lenses of RJMCMC:

- Add a temporary auxiliary variable $\theta^\star \sim G\_0$
  - If the customer was sitting by themselves, augment the state space before the move,
  - Otherwise, augment the state space after the move
- As with the parameters $\lambda$ in the example of lecture 11, the Jacobian is one.

Details: assignment 2

<!--
- Generalized version: "algorithm 8"
- Permanent vs. temporary auxiliary variables
-->
