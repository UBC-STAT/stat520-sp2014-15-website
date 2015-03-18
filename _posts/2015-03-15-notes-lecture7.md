---
layout: post
title: "Lecture 7: Sequential Monte Carlo"
category: 'Lecture'
---

### Review of importance sampling (IS)

- IS with known normalization constants.
- IS without the normalization constants (*self-normalizing IS*).

### Sequential Monte Carlo (SMC) on product spaces

#### Examples 

Let us start with some examples where SMC is useful in Bayesian statistics:

- State space models from assignment 1:
  - $n$: time index (day)
  - Observed number of text messages: $y\_{1:n} = (y\_1, \dots, y\_n)$, $y\_i \in \\{0, 1, 2, \dots\\}$
  - Latent category $x\_{1:n}$, $x\_i \in \\{0, 1\\}$ (note: $x$ was denoted $z$ earlier on)
- Genomics:
  - $n$: positions on a chromosome.
  - Observed single nucleotide polymorphisms (SNP): $y\_{1:n}$, $y\_i \in \\{\textrm{AA}, \textrm{Aa}, \textrm{aa}\\}$
  - Latent *haploblock*. An haploblock is a chunk of the genome with SNP states shared by several individuals. Since there are not too many recombinations, there are well documented haploblocks available for each position $i$, $z\_i \in E\_i$, where $E\_i$ is some discrete set.
- Ultrametric (clock) phylogenetic trees.
  - Species: $S = \\{1, 2, \dots, s\\}$.
  - $E\_i$ contains a partition of $S$ into $s-i$ blocks (to encode the topology of the tree after the $i$-th speciation event), and a real number (the speciation time).

**Common feature:** the latent space is a product space $F\_t = E\_1 \times E\_2 \times \dots \times E\_t$ indexed by the integers $t\in\\{1, \dots, n\\}$. 

**Notes:**

- We may only care about the probability $\pi = \pi\_n$ defined on $F = F\_n$, called the target; the other ones ($t < n$) are called intermediate. 
- This setup was historically the motivation for SMC methods. 
- However, it was discovered in the 2000's that SMC also applies to situations where this is not the case. But let us start by assuming $F$ is a product space, we will get to the general construction later on.

#### Notation and goal

*Target distribution*, with density $\pi(x) = \gamma(x)/Z$ (note: $Z$ was denoted $C$ last time; $x$ was $z$)

**Goals:**

- Computing $Z$ (e.g. to perform model selection)
- Computing $\int f(x) \pi(x) \ud x$, where $f$ is a *test function*.

For example, in the context of Bayesian inference, $Z = m(\textrm{data})$, the marginal likelihood studied last week, and if $f\_i(x\_{1:t}) = x\_i$, then $\int f\_i(x) \pi\_{1:t}(x) \ud x$ is the posterior mean $\E[X\_i | Z\_\{1:t}]$.

#### Sequential Importance Sampling (SIS)

Based on two simple identities:

\\begin{eqnarray}
\gamma(x\_{1:n}) = \frac{\gamma(x\_{1:n})}{\gamma(x\_{1:n-1})} \frac{\gamma(x\_{1:n-1})}{\gamma(x\_{1:n-2})} \dots \frac{\gamma(x\_{1:1})}{\gamma(x\_{\emptyset})},
\\end{eqnarray}

and 

\\begin{eqnarray}
q(x\_{1:n}) = q(x\_1 | x\_\emptyset) q(x\_2 | x\_{1:1}) q(x\_3 | x\_{1:2}) \dots q(x\_n | x\_{1:n-1}),
\\end{eqnarray}

we can write an iterative version of importance sampling, where at each iteration $t = 1, 2, \dots, n$, we carry a *population* of *particles* $x\_{1:t}^1, \dots, x\_{1:t}^N$ with corresponding unnormalized weights $w\_t^1, \dots, w\_t^N$ (see slides).

In SIS, we propose incrementally:

\\begin{eqnarray}
x\_t^i &\sim q(\cdot | x\_{1:t-1}^i) \\\\
x\_{1:t} &= (x\_{1:t-1}^i, x\_t^i),
\\end{eqnarray}

and update the weights using:

\\begin{eqnarray}
w\_t^i &= w\_{t-1}^i \frac{\gamma(x\_{1:t})}{\gamma(x\_{1:t-1})} \frac{1}{q(x\_t | x\_{1:t-1})}.
\\end{eqnarray}

**Exercise:** compute the weights at the last iteration, $w\_n^i$. What is the implication?

The sequential nature of the particle recursions makes it tempting to use SIS in an online setting. This is a bad idea! As we will see, the approximation fails exponentially fast in the number of time steps $n$. Symptom: all the normalized weights converge to 0 except for 1 particle which takes all the normalized weight.

**Examples of weight updates:** 

- HMM. (exercise)
- A coalescent model.

