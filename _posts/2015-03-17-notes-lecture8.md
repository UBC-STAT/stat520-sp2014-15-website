---
layout: post
title: "Lecture 8: SMC samplers and PMCMC"
category: 'Lecture'
---
Instructor: Alexandre Bouchard-C&ocirc;t&eacute;   
Editor: Creagh Briercliffe


#### Sequential Monte Carlo (SMC)

**Motivation:** to fix or at least alleviate the issue raised at the end of the [last section](http://www.stat.ubc.ca/~bouchard/courses/stat520-sp2014-15/lecture/2015/03/15/notes-lecture7.html). Namely, in Sequential Importance Sampling (SIS) all of the normalized weights converge to 0, except for 1 particle which takes all of the normalized weight.

**Intuition:** prune particles of low weights between each SIS iteration (or when the population becomes too unbalanced).

**Constraint:** we would like to do our pruning in such a way that the consistency of Importance Sampling (IS) is preserved. That is, we would like to be able to (a.s.) approach both $Z$ (the normalizing constant) and $\int f(x) \pi(x) \ud x$ as the number of particles goes to infinity.

**Solution:** Resampling will be a way to prune particles while preserving consistency. The simplest scheme is called *multinomial resampling*. We explain it via an example.

**Multinomial resampling:** 

- Suppose we are given an urn containing balls of different colors. 
- This urn can be viewed as a probability distribution over colors.
- To approximate this probability distribution, I draw 100 times with replacement and assign to each color the fraction of times I drew this color.
- More abstractly, this process can take any probability measure as an input, and create an approximation distribution which can be viewed as 100 equally weighted particles.

In SMC, the balls are the particle population from the previous generation, and instead of 100 draws we pick the number of particles as the number of resampling draws.

**Theoretical challenge:** this creates interaction/dependencies across particles, making it harder to prove consistency than in the IS or SIS setups.

**Other pictures for multinomial resampling:** throwing darts on a colored stick. Based on where the dart lands, assign to each color the fraction of times the darts land on this color.

**Lower variance resampling alternatives:** 

- *Stratified resampling*: split the colored stick into bins of equal size and throw one dart uniformly in each bin.
- *Systematic resampling*: same as stratified resampling, but we reuse the 100 uniform random numbers &mdash; thereby deterministiaclly linking all of the samples in each bin.
- For more, see: [http://biblio.telecom-paristech.fr/cgi-bin/download.cgi?id=5755](http://biblio.telecom-paristech.fr/cgi-bin/download.cgi?id=5755).

**Note:** only use resampling when it is really needed, i.e. when there are indeed many low weight particles in the current iteration. A useful strategy is to monitor the effective sampling size (ESS) at each iteraion, given by the following.

\\begin{eqnarray}
ESS := \frac{\left(\sum\_{i=1}^N w^i\right)^2}{\sum\_{i=1}^N (w^i)^2 },
\\end{eqnarray}

which is maximized to $N$ if all weights are equal ($1/N$), and minimized to $1$ if one particle has all of the mass. The strategy is to resample only when the ESS is smaller than $N/2$.

#### Another view of SMC

Alternatively, we can view SMC as an algorithm using importance sampling to first approximate $\pi\_1$, then approximate $\pi\_2$, etc. Let us denote these *intermediate approximations* by $\tilde \pi\_t$ for the $t^{th}$ approximation.

**Question:** what proposal should we use?

**Idea:** to construct the proposal, use $\tilde \pi\_{t-1}$ and a *transition* proposal $q(x\_t|x\_{t-1})$, where we assume that we can sample from $q$ and evaluate $q(x\_t|x\_{t-1})$ with the correct normalization.

**Details:** [See slides](http://www.stat.ubc.ca/~bouchard/courses/stat520-sp2014-15/files/mar16.pdf).



### SMC Samplers: SMC on general spaces

We lift the assumptions that $F\_t = E\_1 \times E\_2 \times \dots \times E\_t$. Let us now assume that $F\_t$ is an arbitrary space.

**Motivations:**

- Phylogenetic inference on non-clock trees. See the [slides](http://www.stat.ubc.ca/~bouchard/courses/stat520-sp2014-15/files/mar16.pdf) and [The Phylogenetic Handbook](http://www2.ib.unicamp.br/profs/sfreis/SistematicaMolecular/Aula08MetodosMatrizesDistancias/Leituras/ThePhylogeneticHandbookMatrizesDistancias.pdf) by Lemey, Salemi, & Vandamme.
- Annealing, $\pi\_t = p(x) (\ell(y | x))^{\alpha\_t}$, where $\alpha\_t$ is increasing to $\alpha\_n = 1$. See [Annealed Importance Sampling](http://www.cs.toronto.edu/~radford/ais-pub.abstract.html) by Radford Neal.

**Why do we need special consideration for non-product space?** Example: overcounting in discrete state spaces. See the [slides](http://www.stat.ubc.ca/~bouchard/courses/stat520-sp2014-15/files/mar16.pdf) for an example where there is more than one way to build the same tree structure.

**Solution:** For the purpose of analysis, we build auxiliary spaces and distributions as follows. 

- $\tilde F\_t = F\_1 \times F\_2 \times \dots \times F\_t$ 
- construct a *backward* transition model, $q^-(x\_{t-1}|x\_t)$ for $x\_i \in F\_i$ ([more on that soon](http://www.stat.ubc.ca/~bouchard/courses/stat520-sp2014-15/lecture/2015/03/22/notes-lecture9.html))
- for any $x\_{1:t}$, $x\_i \in F\_i$, set $\tilde \gamma\_t(x\_{1:t})$ as follows:

\\begin{eqnarray}
\tilde \gamma\_t(x\_{1:t}) = \gamma\_t(x\_t) q^-(x\_{t-1}|x\_t) q^-(x\_{t-2}|x\_{t-1}) \dots q^-(x\_{1}|x\_2)
\\end{eqnarray}

The resulting framework is called a SMC *sampler*. For more information see Section 3 of the [Sequential Monte Carlo Samplers](http://www.stats.ox.ac.uk/~doucet/delmoral_doucet_jasra_sequentialmontecarlosamplersJRSSB.pdf) paper by Del Moral, Doucet & Jasra.

**Exercises:**

- Marginalize $\tilde \pi\_t(x\_{1:t})$ over $x\_{1:t-1}$, where $\tilde \pi\_t \propto \tilde \gamma\_t$.
- Since $\tilde \gamma\_t$ is now defined over a product space $\tilde F\_t$, we can use standard SMC on that auxiliary construction. Find the weight update of a standard SMC algorithm targeting $\tilde \gamma\_t$.



