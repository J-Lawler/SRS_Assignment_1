########## Simulate Data #############

set.seed(31)

n <- 1000

# True Parameters

# parameters for g
phi1_true <- 0.98
sigma_true <- 0.2
eta1 <- rnorm(n)
# parameters for h
phi2_true <- 0.5
eta2 <- rnorm(n)
# parameters for y
beta_true <- 0.05
epsilon <- rnorm(n)

# initial values
g_1 <- 1
h_1 <- 1

# state process
g_values <- numeric(n)
g_values[1] <- g_1

h_values <- numeric(n)
h_values[1] <- h_1

# state-dependent process
y_values <- numeric(n)
y_values[1] <- epsilon[1]*beta_true*exp((g_1+h_1)/2)

for(i in 1:(n-1)){
  g_values[i+1] = phi1_true*g_values[i] + sigma_true*eta1[i]
  h_values[i+1] = phi2_true*h_values[i] + sigma_true*eta2[i]
  y_values[i+1] <- epsilon[i+1]*beta_true*exp((g_values[i+1]+h_values[i+1])/2)
}


########## HMM Method #############


loglikelihood.SV0 <- function(parameters, y, m, gmax){
  # Unpack parameter vector
  phi1 <- parameters[1]
  sigma <- exp(parameters[2]) # constraining sigma to be positive
  beta <- exp(parameters[3]) # constraining beta to be positive
  phi2 <- parameters[4]
  
  K <-  m+1
  # Compute the interval endpoints. 
  bg <-  seq(-gmax,gmax,length=K) # for g
  bh <-  seq(-gmax,gmax,length=K) # for h
  # Compute the interval midpoints.
  bmid_g <-  (bg[-1]+bg[-K])*0.5
  bmid_h <-  (bh[-1]+bh[-K])*0.5
  
  midpoint_combinations <- expand.grid(bmid_g,bmid_h)
  # Compute the std deviation of yt for each interval midpoint.
  sey <-  beta*exp((midpoint_combinations[,1]+midpoint_combinations[,2])/2)
  Gamma <-  matrix(0,m^2,m^2)
  for (i in 1:m^2){
    # Compute the t.p.m. Γ using expression (5).
    Gamma[i,] <- diff(pnorm(bg,phi1*midpoint_combinations[i,1],sigma)* pnorm(bh,phi2*midpoint_combinations[i,2],sigma))} # product due to independence
  
  # Scale the rows of Γ so that they each sum to 1.
  Gamma <-  Gamma/apply(Gamma,1,sum)
  # Compute δ(=δΓ), the stationary distribution of the Markov chain {ht}.
  # This will be our initial distribution
  foo <-  solve(t(diag(m^2)-Gamma+1),rep(1,m^2)) 
  
  llk <-  0
  # Loop to compute the log-likelihood, i.e. the log of expression (6).
  for (t in 1:length(y)){
    foo <-  foo%*%Gamma*dnorm(y[t],0,sey) 
    sumfoo <- sum(foo)
    llk <-  llk+log(sumfoo) 
    foo <-  foo/sumfoo}
  # Return negative log likelihood since optim() minimises
  return(-llk)}



test_parameters <- c(0.9,log(0.3),log(0.2),0.9)


#loglikelihood.SV0(test_parameters, y_values, 40, 4)



start_time_HMM <- Sys.time()

# minimise likelihood
HMM_estimates <- nlm(loglikelihood.SV0, test_parameters,
                     y=y_values,  m = 30, gmax = 4)

end_time_HMM <- Sys.time()

end_time_HMM - start_time_HMM


## HMM Results

# phi1
HMM_estimates$estimate[1]

# sigma
exp(HMM_estimates$estimate[2])

# beta
exp(HMM_estimates$estimate[3])

# phi2
HMM_estimates$estimate[4]