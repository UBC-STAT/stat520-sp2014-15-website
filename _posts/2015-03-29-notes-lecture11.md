---
layout: post
title: "Lecture 11: From model selection to Bayesian non-parametrics"
category: 'Lecture'
---

#### Prerequisites

- Two simulation concepts in Bayesian statistics: recall that a Bayesian model specifies a joint distribution on the known $Y$ and unknown $X$.
   - Posterior simulation: $X \mid Y$
   - Forward simulation: $(X, Y)$.
- Today, we will focus on mixture models, which is an important source of Bayesian non-parametric models (but not the only one!)


#### Model selection vs. Bayesian non-parametric models

- We will contrast forward simulation in the two models.
- Let $P\_n$ denote the number of parameters used to generate $n$ datapoints ($\propto$ the number of distinct clusters)
- Let $P = \lim\_{n\to \infty} P\_n$ 
- Note: $P\_n$ are $P$ are random!
- Distinction:
   - Bayesian parametric model: $\P(P = \infty) = 0$
   - Bayesian non-parametric model (BNP): $\P(P = \infty) > 0$.
- Examples of applications where model selection/BNP is preferred:
   - Model selection: clustering, cases where the parametric component is a good approximation.
   - BNP: density estimation, especially when the parametric model is not trusted. Phenomena with fat tails, e.g. frequency of words (do you think assuming the number of existing words is finite is a good idea?)
   
However, note that for posterior simulation, the number of parameters is bounded by the number of datapoints, therefore similar MCMC techniques can be used in both cases (in particular, RJMCMC probably have been underused in BNP posterior inference).

#### Viewing clustering priors as random probability distributions

- Let us say we are given a set  $A \subset \RR \times (0, \infty)$  <img src="{{ site.url }}/images/setA.jpg" alt="Drawing" style="width: 200px; float: right"/>
  - The space $\RR \times (0, \infty)$ comes from the fact that we want $\theta$ to have two components, a mean and a variance, where the latter has to be positive
  - Let us call this space $\Omega = \RR \times (0, \infty)$
- We can define the following probability mesure <img src="{{ site.url }}/images/dirichletRealization.jpg" alt="Drawing" style="width: 200px; float: right"/>
\\begin{eqnarray}
G(A) = \sum\_{k=1}^{K} \pi\_k \1(\theta\_k \in A),
\\end{eqnarray} 
where the notation $\1(\textrm{some logical statement})$ denotes the indicator variable on the event that the boolean logical statement is true. In other words, $\1$ is a random variable that is equal to one if the input outcome satisfies the statement, and zero otherwise.

**Random measure:** Since the location of the atoms and their probabilities are random, we can say that $G$ is a random measure. The Dirichlet distribution, together with the prior $p(\theta)$, define the distribution of these random discrete distributions.

**Another view of a random measure:** a collection of real  random variables indexed by sets in a sigma-algebra: $G\_{A} = G(A)$, $A \in \sa$.

**Limitation of our setup so far:** we have to specify the number of components/atoms $K$. The Dirichlet *process* is also a distribution on atomic distributions, but where the number of atoms can be countably infinite.

#### Dirichlet processes and other random probability distributions

**Definition:**  A Dirichlet process (DP), $G\_A(\omega) \in [0, 1]$, $A \in \sa$, is specified by:

1. A *base measure* $G\_0$ (this corresponds to the density $p(\theta)$ in the previous example).
2. A *concentration parameter* $\alpha\_0$ (this plays the same role as $\alpha + \beta$ in the simpler Dirichlet distribution).

To do forward simulation of a DP, do the following: <img src="{{ site.url }}/images/dpSimulation.jpg" alt="Drawing" style="width: 500px; float: right"/>

1. Start with a current stick length of one: $s = 1$
2. Initialize an empty list $r = ()$, which will contain atom-probability pairs.
3. Repeat, for $k = 1, 2, \dots$
   1. Generate a new independent beta random variable: $\beta \sim \Beta(1, \alpha\_0)$.
   2. Create a new stick length: $\pi\_k = s \times \beta$.
   3. Compute the remaining stick length: $s \gets s \times (1 - \beta)$
   4. Sample a new independent atom from the base distribution: $\theta\_k \sim G_0$.
   5. Add the new atom and its probability to the result: $r \gets r \circ (\theta\_k, \pi\_k)$

Finally, return:
\\begin{eqnarray}
G(A) = \sum\_{k=1}^{\infty} \pi\_k \1(\theta\_k \in A),
\\end{eqnarray} 

This is the *Stick Breaking representation*, and is not quite an algorithm as written above (it never terminates), but this can be fixed by lazy computing.

**Other examples:**

- Pitman-Yor. Make the parameters of the beta depend on the level (at level $i$, use $(1-d, \alpha + id)$, where $d$ is a tuning parameter $d \in [0, 1)$.
- Normalized random measures (in particular, normalized generalized gamma, which includes Normalized inverse gaussian, DP, normalized stable).
- Most general category, Exchangeable Partition Probability Functions (EPPF) not believed to be tractable.

Key motivation for these extensions: DPs can give us $P\_n \sim \log(n)$. PY gives us $P\_n \sim n^d$. PY also yields a power law distribution on the number of customers per table (i.e. the number $m$ of customers is proportional to $m^{-(1+d)}$ (truncated to avoid the asymptote at zero).

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

\\begin{eqnarray}
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