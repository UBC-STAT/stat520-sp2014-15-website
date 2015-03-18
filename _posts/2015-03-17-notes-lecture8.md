---
layout: post
title: "Lecture 8: SMC samplers and PMCMC"
category: 'Lecture'
---

#### SMC

**Motivation:** fix or at least alleviate the issue raised at the end of the last section.

**Intuition:** prune particles of low weights between each SIS iteration (or when the population becomes too unbalanced).

**Constraint:** we would like to do our pruning in such a way that the consistency of IS is preserved. I.e. we would like to be able to approach (a.s.) both $Z$ and $\int f(x) \pi(x) \ud x$ as the number of particles goes to infinity.

**Resampling:** a way to prune while preserving consistency. The simplest scheme is called multinomial resampling. We explain it via an example:

**Multinomial resampling:** 

- Suppose we are given an urn containing balls of different colors. 
- Note: this urn can be viewed as a probability distribution over colors.
- To approximate this probability distribution, I draw 100 times with replacement and assign to each color the fraction of times I drew this color.
- More abstractly, this process can take any probability measure as an input, and create an approximation distribution which can be viewed as 100 equally weighted particles.

In SMC, the input is the particle population from the previous generation, and instead of 100 we pick the number of particles as the number of resampling draws.

**Theoretical challenge:** this creates interaction/dependencies across particles, making it harder to prove consistency than in the IS or SIS setups.

**Other picture for multinomial resampling:** throwing darts on a colored stick.

**Lower variance resampling alternatives:** 

- Stratified resampling: split the colored stick into bins of equal size and sample one dart uniformly in each bin.
- Systematic resampling: same as stratified, but reuse the 100 uniform random numbers.
- For more, see: [http://biblio.telecom-paristech.fr/cgi-bin/download.cgi?id=5755](http://biblio.telecom-paristech.fr/cgi-bin/download.cgi?id=5755).

#### Another view on SMC

We can alternatively view SMC as an algorithm using importance sampling to first approximate $\pi\_1$, then to approximate $\pi\_2$, etc. Let us call these *intermediate approximations*, $\tilde \pi\_t$.

**Question:** what proposal to use?

**Idea:** to construct the proposal, use $\tilde \pi\_{t-1}$ and a *transition* proposal $q(x\_t|x\_{t-1})$ (where we assume we can sample from $q$ and evaluate $q(x\_t|x\_{t-1})$ with the correct normalization.

**Details:** See slides.


### Sequential Monte Carlo (SMC) on general spaces

We lift the assumptions that $F\_t = E\_1 \times E\_2 \times \dots \times E\_t$. Let us assume now that $F\_t$ is an arbitrary space.

**Motivations:**

- Phylogenetic inference on non-clock trees.
- Annealing, $\pi\_t = p(x) (\ell(y | x))^{\alpha\_t}$, where $\alpha\_t$ is increasing, $\alpha\_n = 1$.

**Why we need special consideration for non-product space?** Example: overcounting in discrete state spaces.

**Solution:** For the purpose of analysis, build auxiliary spaces and distributions as follows 

- $\tilde F\_t = F\_1 \times F\_2 \times \dots \times F\_t$ 
- construct a *backward* transition model (more on that soon), $q^-(x\_{t-1}|x\_t)$ for $x\_i \in F\_i$,
- for any $x\_{1:t}$, $x\_i \in F\_i$, set $\tilde \gamma\_t(x\_{1:t})$ as follows:

\\begin{eqnarray}
\tilde \gamma\_t(x\_{1:t}) = \gamma\_t(x\_t) q^-(x\_{t-1}|x\_t) q^-(x\_{t-2}|x\_{t-1}) \dots q^-(x\_{1}|x\_2)
\\end{eqnarray}

**Exercises:**

- Marginalize $\tilde \pi\_t(x\_{1:t})$ over $x\_{1:t-1}$, where $\tilde \pi\_t \propto \tilde \gamma\_t$.
- Since $\tilde \gamma\_t$ is now defined over a product space, we can use standard SMC on that auxiliary construction. Find the weight update of a standard SMC algorithm targeting $\tilde \gamma\_t$.

<!--
**Estimate of $Z$:**


### Overview of some theoretical properties

Here we look informally at the main theoretical properties enjoyed by SMC. We will come back for a more detailed treatment next lecture.

#### (Algorithmic) consistency



- why proofs are non-trivial: interaction and dependence

#### Bias calculations

- nb: less important

#### Variance calculations

#### Asymptotic normality

### Generic sequences of target distributions

- Annealing
- Phylogenetic trees

- auxiliary construction
-->