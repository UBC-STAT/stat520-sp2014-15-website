---
layout: post
title: "Lecture 6: Model choice, continued"
category: 'Lecture'
---

### Review

We saw last time that a central goal in Bayesian model selection is to compute marginal likelihoods $m(x)$ and Bayes factors $B\_{12}$. This is generally not as direct as computing posterior expectations, especially when both continuous and discrete distributions are involved. We talked about two methods last time: conjugate calculations, and model saturation, each with their own pros and cons. Today, we will cover more methods.

### Parenthesis: project idea

Given the complexity of Bayesian model selection method when both continuous and discrete distributions are involved, it is generally a good idea to try to reduce the problem to a fully discrete case. How to do this? Analytic marginalization. Why? Opportunity for new models..

### Bayes factor, continued

#### Implementation of model saturation

Example on the Poisson process models from last time.

#### Importance sampling and the harmonic estimator

We now consider the problem of evaluating the marginal likelihood of a single model (instead of a ratio). We start by looking at the more general issue of approximating integral using importance sampling.

Let $\pi(z)$ denote a density of interest, and $f(z)$, any (integrable) function. our goal is to approximate $\int f(z) \pi(z) \ud z$.

For example, we will take $f(z) = \ell(x | z)$, and $\pi(z)$ to be the prior here, since with this choice:

\\begin{eqnarray}
m(x) &= \int f(z) \pi(z) \ud z.
\\end{eqnarray}

Suppose that $q(z)$ is a density with a support containing the support of $\pi(z)$. We assume that we can sample from $q$, or that we can sample from a Markov chain with stationary distribution $q$. $q$ is called the proposal distribution.

Moreover, assume to start with that we can compute $q(z)$ and $\pi(z)$ for any $z$ (including the normalization). We will relax this assumption.

Informally, the key idea behind importance sampling is: (1) to divide and multiply by $q(z)$:

\\begin{eqnarray}
\int \;\; f(z) \pi(z) \ud z = \int \;\; f(z) \pi(z) \frac{q(z)}{q(z)} \ud z,
\\end{eqnarray}

(2), to re-interpret the above equation as being an expectation with respect to $q$:

\\begin{eqnarray}
\int \;\; f(z) \pi(z) \frac{q(z)}{q(z)} \ud z &= \int f(z) w(z) q(z) \ud z \\\\
&= \E[w(\tilde Z) f(\tilde Z)],
\\end{eqnarray}

where $\tilde Z \sim q$ and $w(z) = p(z)/q(z)$, and (3), use the law of large numbers (LLN) to justify the following approximation:

\\begin{eqnarray}
\E[w(\tilde Z)] \approx \frac{1}{N} \sum_{i = 1}^N f(\tilde Z^{(i)}) w(\tilde Z^{(i)}),
\\end{eqnarray}

where $\tilde Z^{(i)}$ could come from:

- iid samples from $q$ (justified using the standard LLN), or
- samples from a Markov chain with stationary density $q$ (Markov chain LLN).

---

**Back to the marginal likelihood:** a first choice would be $q(z) = p(z)$ (recall that we pick $f(z) = \ell(x|z)$ throughout this section). In this case, we get:

\\begin{eqnarray}
\frac{1}{N} \sum_{i = 1}^N \ell(x | z^{(i)}),
\\end{eqnarray}

however this does not work well in practice since the prior and posterior can be quite different. 

To improve on this, we will need a second version of importance sampling that does not require having access to the normalized density $q$.

---

Let us now relax the assumption that the normalization of both $q$ and $p$ are known. Let $\pi(z) = \gamma(z)/C\_{\pi}$, where $\gamma(z)$ is easy to compute, but $C\_{\pi}$ is hard. This will take care of the normalization of $q$ at the same time.

Now use the same idea as above (1-3) to approximate:

\\begin{eqnarray}
C\_{\pi} = \int \gamma(z) \ud z,
\\end{eqnarray}

obtaining the approximation

\\begin{eqnarray}
C\_{\pi} \approx \frac{1}{N} \sum\_{i = 1}^N w'(\tilde Z^{(i)}),
\\end{eqnarray}

where $w'(z) = \gamma(z) / q(z)$.

Finally, invoking Slutsky justifies the approximation:

\\begin{eqnarray}
\frac{\frac{1}{N} \sum\_{i = 1}^N f(\tilde Z^{(i)}) w'(\tilde Z^{(i)})}{\frac{1}{N} \sum\_{i = 1}^N w'(\tilde Z^{(i)})} &= \frac{\sum\_{i = 1}^N f(\tilde Z^{(i)}) w'(\tilde Z^{(i)})}{ \sum\_{i = 1}^N w(\tilde Z^{(i)})} \\\\
&= \sum\_{i = 1}^N f(\tilde Z^{(i)}) \bar w'(\tilde Z^{(i)}),
\\end{eqnarray}

where $\bar w$ denote the normalized weights.

Note also that in this final equation, $q(z)$ also appears on both the numerator and denominator. Therefore, the approximation works with $w''(z) = \gamma(z) / u(z)$, where $q(z) = u(z) / C\_q$.

---

**Back to the marginal likelihood:** Let us now take $u(z) = \ell(x|z) p(z)$, with again $f(z) = \ell(x|z)$. We get:

\\begin{eqnarray}
\left( \frac{1}{N} \sum\_{i=1}^N (\ell(x|z^{(i)}))^{-1} \right)^{-1}
\\end{eqnarray}

Unfortunately, the variance of this estimator is often infinite, resulting in very inefficient estimates.

**Pro:** easy to implement, does not require ratios.

**Con:** does not work reliably in practice. Avoid using this outside of a baseline for  benchmarking other more sophisticated methods.

#### Bridge sampling

**Exercise:** assuming $\pi\_i(z) = \gamma\_i(z)/C\_i$ for $i \in \\{0, 1\\}$, compute:

\\begin{eqnarray}
\frac{\E[\gamma\_0(Z\_1) h(Z\_1)]}{\E[\gamma\_1(Z\_0) h(Z\_0)]},
\\end{eqnarray}

where $Z\_i \sim \pi\_i$,  $h$ is an arbitrary function called a *bridge function*. A Monte Carlo estimator can be derived easily from the result of your calculation if $Z\_i^{(j)}$ are distributed according to $\pi\_i$ (or have $\pi\_i$ as their stationary distribution in the MCMC case).

**Note:** this method is specialized to cases where the two models $\pi\_0(z)$ and $\pi\_1(z)$ are defined over the same space. One could in theory take $\pi\_0$ to be the prior $p$, but in this case the method reduces to importance sampling.

#### Thermodynamic integration

As in bridge sampling, we target the ratio $B\_{12}$, but we consider a continuum of intermediate distributions between $\pi\_1$ and $\pi\_2$. This opens the door to taking $\pi\_0$ to be the prior $p$, with intermediate distributions $\pi\_t = \gamma\_t / C\_t$, $t\in[0,1]$ given by $\gamma\_t(z) = p(z) (\ell(x|z))^t$.

We start with the following standard identity, which holds under the assumption that we can swap a derivative and an integral (see for example Folland, Real Analysis):

\\begin{eqnarray}\label{eq:thermo}
\frac{\ud}{\ud t} \log C\_t = \E\_t\left[ \frac{\ud}{\ud t} \log \gamma\_t(Z) \right],
\\end{eqnarray}

where $\E\_t$ denotes expectation with respect to $Z \sim \pi\_t$. We define $U\_t(z) = \frac{\ud}{\ud t} \log \gamma\_t(z)$.

Thermodynamic integration consists in integrating both sides of Equation (\ref{eq:thermo}) from 0 to 1, yielding:

\\begin{eqnarray}
\log C\_1 - \log C\_0 = \int\_0^1 \E\_t[U\_t(Z)] \ud t.
\\end{eqnarray}

This can be approximated for example by numerical integration of the univariate integral, and Monte Carlo integration in the inner loop; or by using Monte Carlo on an auxiliary space including $t$ as a random variable.

**Pro:** generally more accurate then the other methods.

**Con:** generally more computationally expensive. 

#### Nested sampling

This method computes $C = m(x)$ from our usual Bayesian setup, $\pi(z|x) = p(z) \ell(x|z) / C$. Since the observation $x$ is fixed, let us abbrivate $\ell(x|z)$ by $\ell(z)$.

We will rewrite $C$ using the following standard formula: if $Y$ is a random variable with cdf $F(y)$, then:

\\begin{eqnarray}
\int\_0^\infty (1 - F(y)) \ud y = \E[Y].
\\end{eqnarray}

**Example:** write the expectation of a dice in two ways to get some intuition why this is true.

Let us apply this to the problem of computing $C$, which we can view as an expectation under the prior as follows:

\\begin{eqnarray}
C = \int p(z) \ell(z) \ud z = \E\_p \ell(Z),
\\end{eqnarray}

where $Z \sim p$. Therefore, if we set:

\\begin{eqnarray}
G(\lambda) = \int\_{\\{z : \ell(z) > \lambda\\}} \;\;\;p(z) \ud z,
\\end{eqnarray}

then:

\\begin{eqnarray}\label{eq:nested1}
C = \int\_0^\infty\;\; G(\lambda) \ud \lambda.
\\end{eqnarray}

Equivalently:

\\begin{eqnarray}\label{eq:nested2}
C = \int\_0^1\;\; G^{-1}(p) \ud p.
\\end{eqnarray}

(to see why this is equivalent, interpret each of the integrals in Equations (\ref{eq:nested1}) and (\ref{eq:nested2}) as an area under the curve.

Nested sampling is an approximation of the integral obtained via numerical integration on $[0, 1]$ paired with a Monte Carlo approximation scheme to approximate $G^{-1}(p)$. 

Let us start by describing how the value of $G^{-1}(p)$ could be approximated for the largest $p$ in the numerical approximation:

1. Simulate $N = 10$ points $z\_1, \dots, z\_N$ according to the prior $p$.
2. Find the smallest likelihood value $\lambda\_{\textrm{min}} = \min\_i \ell(z\_i)$. Let's say 0.007.
3. This suggests the approximation $G^{-1}(0.9) \approx 0.007$, since $G^{-1}(p) = \lambda$ if and only if $\P(\ell(Z) > \lambda) = p$ under the prior, $Z \sim p$.

Note that as $p$ gets smaller, we will generally want more and more closely spaced grid points. 

So to get the second largest value of $G^{-1}(p)$, we will first add one more point. To make sure it is "useful", let us assume we can simulate one from the truncated prior,

\\begin{eqnarray}
p\_\textrm{trunc}(z) \propto p(z) \1[z > \lambda\_{\textrm{min}}].
\\end{eqnarray}

This will give us an approximation for $G^{-1}\left( \left(\frac{N-1}{N} \right)^{2} \right)$.

**Pro:** can be more accurate then MCMC based methods.

**Con:** sampling from the truncated prior can be difficult.

#### Other methods for estimating Bayes factors

- Reversible jump MCMC.
- SMC samplers
- Chib's method

We might cover some of these later in the course.

### Other model selection ideas

- Frequentist validation.
- Bayesian deviance.
- Classical approximations (BIC).