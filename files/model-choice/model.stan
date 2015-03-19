functions {

real power_normal_log(real y, real mu, real sigma, real t)
{
  return t * normal_log(y, mu, sigma);
}

}

data {  
  real x;
  real<lower = 0, upper = 1> t;
  real<lower = 0 > sigma_mean;
  real<lower = 0 > sigmax;
}
parameters {
  real meanx;
}

model {
  x ~ power_normal(meanx, sigmax, t);
  meanx ~ normal(0, sigma_mean);
}

generated quantities {
  real marginal_t;
  marginal_t <- normal_log(x, meanx, sigmax);
}