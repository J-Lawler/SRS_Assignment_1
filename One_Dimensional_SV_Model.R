
################## Simple Stochastic Volatility Model SV0 ##################

###### Simulate  Data ###### 

set.seed(31)

# True Parameters
phi_true <- 0.98
sigma_true <- 0.2
beta_true <- 0.05
g_1 <- 1

n <- 1000
eta <- rnorm(n)
epsilon <- rnorm(n)

# state process
g_values <- numeric(n)
g_values[1] <- g_1
 
# state-dependent process
y_values <- numeric(n)
y_values[1] <- epsilon[1]*beta_true*exp(g_1/2)

for(i in 1:(n-1)){
  g_values[i+1] = phi_true*g_values[i] + sigma_true*eta[i]
  y_values[i+1] <- epsilon[i+1]*beta_true*exp(g_values[i+1]/2)
}





###### Estimate with HMM Methods ######

# Code adapted from Appendix B of:

# Roland Langrock, Iain L. MacDonald, Walter Zucchini,
# Some nonstandard stochastic volatility models and their estimation using structured hidden Markov models,
# Journal of Empirical Finance,
# Volume 19, Issue 1,
# 2012,
# Pages 147-161,

m <- 200
gmax <- 4

loglikelihood.SV0 <- function(parameters, y, m, gmax){
  # Unpack parameter vector
  phi <- parameters[1]
  sigma <- exp(parameters[2]) # constraining sigma to be positive
  beta <- exp(parameters[3]) # constraining beta to be positive
  
  K <-  m+1
  # Compute the interval endpoints. 
  b <-  seq(-gmax,gmax,length=K)
  # Compute the interval midpoints.
  bs <-  (b[-1]+b[-K])*0.5
  # Compute the std deviation of yt for each interval midpoint.
  sey <-  beta*exp(bs/2)
  Gamma <-  matrix(0,m,m)
  for (i in 1:m){
    # Compute the t.p.m. Γ using expression (5).
    Gamma[i,] <- diff(pnorm(b,phi*bs[i],sigma))}
  
  # Scale the rows of Γ so that they each sum to 1.
  Gamma <-  Gamma/apply(Gamma,1,sum)
  # Compute δ(=δΓ), the stationary distribution of the Markov chain {ht}.
  # This will be our initial distribution
  foo <-  solve(t(diag(m)-Gamma+1),rep(1,m)) 
  
  llk <-  0
  # Loop to compute the log-likelihood, i.e. the log of expression (6).
  for (t in 1:length(y)){
    foo <-  foo%*%Gamma*dnorm(y[t],0,sey) 
    sumfoo <- sum(foo)
    llk <-  llk+log(sumfoo) 
    foo <-  foo/sumfoo}
  # Return negative log likelihood since optim() minimises
  return(-llk)}

start_time_HMM <- Sys.time()

# minimise likelihood
HMM_estimates <- nlm(loglikelihood.SV0, c(0.9,log(0.3),log(0.2)),
            y=y_values,  m = m, gmax = gmax)

end_time_HMM <- Sys.time()

end_time_HMM - start_time_HMM


# phi
HMM_estimates$estimate[1]

# sigma
exp(HMM_estimates$estimate[2])

# beta
exp(HMM_estimates$estimate[3])


###### Estimate with HMC & Stan ######


# code adapted from Stan user manual
# https://mc-stan.org/docs/stan-users-guide/stochastic-volatility-models.html

library(tidyverse)
library(tidybayes)
library(rstan)

stan_code <- 
  "data{
    vector[1000] y_values;
}
parameters{
    real phi;
    real<lower=0> sigma; 
    real<lower=0> beta;
    vector[1000] g;
}
model{
    phi ~ uniform(-1, 1); // prior for phi
    sigma ~ cauchy(0, 5); // prior for sigma
    beta ~ cauchy(0, 5); // prior for beta
    g[1] ~ normal(0, sigma / sqrt(1 - phi * phi)); // prior for initial value
    for (t in 2:1000) {
    g[t] ~ normal(phi *g[t - 1], sigma);
    }
    for (t in 1:1000) {
    y_values[t] ~ normal(0, beta*exp(g[t] / 2));
    }
}"

start_time_sample <- Sys.time()

# stan_model compiles the model - the data is not used and no samples are drawn
stan_sv0 <- stan_model(model_name = "stan_sv0",model_code=stan_code)

# compose_data is a tidybayes function
# sampling draws samples from the posterior
model_samples <-   sampling(stan_sv0, data = compose_data(y_values), chains=2)

# saveRDS(stan_sv0,file="stan_sv0.rds")
# stan_sv0 <- readRDS(file="stan_sv0.rds")

# The output is now a stanfit object that contains the draws from the posterior

end_time_sample <- Sys.time()

end_time_sample - start_time_sample

# saveRDS(stan_sv0,file="stan_sv0.rds")
# stan_sv0 <- readRDS(file="stan_sv0.rds")

# The output is now a stanfit object that contains the draws from the posterior


# Extracting Draws
stan_draws <- model_samples %>% 
  spread_draws(phi, sigma, beta)

mean(stan_draws$phi)
mean(stan_draws$sigma)
mean(stan_draws$beta)






