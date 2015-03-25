require(rstan)
require(ggplot2)
require(caTools)
require(coda)
source("http://mc-stan.org/rstan/stan.R")

true.marg.log.lik <- function (x, s0, s1)
{ 
  dnorm(x, 0, sqrt(s0 ^ 2 + s1 ^ 2), log = T)
}

createMCMC <- function(x, data)
{
  d <- mcmc(data = data[ , , x])
  return(d)
}

stan2coda <- function(fit) {
  a <- as.array(fit)
  nVars <- dim(a)[3]
  coda <- mcmc.list(lapply(1:nVars, createMCMC, a))
  names(coda) <- names(a[1, , ])
  return(coda)
}

## Global variables
nIters <- 100000

## Generate data (uncomment as required)
set.seed(123123)
s1 <- 1
s0 <- 2
meanx <- rnorm(1, 0, s0)
x <- rnorm(1, meanx, s1)

t <- seq(0, 1, length = 20)
marginal <- matrix(0, nrow = length(t), ncol = 1)

for (i in 1:length(t))
{
  df <- list(x = x, t = t[i], sigmax = s1, sigma_mean = s0)
  fit <- stan(file = 'model.stan', data = df, iter = nIters, chains = 1, seed = 1, thin = 10)
  samples <- stan2coda(fit)
  marginal[i]  <- mean(samples$marginal_t)  
}

estimateZ <- trapz(t, marginal)
marginalLik <- data.frame(t, logLik = marginal)
shade <- rbind(c(0,0), marginalLik, c(marginalLik[nrow(marginalLik), "t"], 0))
p <- ggplot(data = marginalLik, aes(x = t, y = logLik)) + 
  geom_point() + 
  geom_line() + 
  annotate("text", x = 0.5, y = -2.25, label = paste("Area under curve approximately ", round(estimateZ, 2), sep = "")) + 
  annotate("text", x = 0.5, y = -2.5, label = paste("True marginal log-likelihood ", round(true.marg.log.lik(x, s0, s1), 2), sep = "")) + 
  geom_polygon(data = shade, aes(x = t, y = logLik), alpha = 0.5) +
  theme_bw() 

ggsave("TI_neal_example.png", plot = p, width = 11, height = 8.5)

