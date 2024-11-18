library(tidyverse)
library(ggplot2)
library(cmdstanr)
library(ggthemes)
theme_set(theme_few())


# loading estimator script
source("estimator/estimate.r")

#################################
######## Data Generation ########
#################################
# Simulation parameters
N <- 100 # units
T <- 50 # time periods
tau <- 0.5 # treatment effects

# Generating Data 
# adoption times and treatment matrix(staggered adoption)
adoption_times <- sample(1:T, N, replace = TRUE)
W <- matrix(0, N, T)
for (n in 1:N) {
  W[n, adoption_times[n]:T] <- 1
}
# outcome matrix
Y <- matrix(0, N, T)
Y <- Y + (tau*W)


# running CV
results <- estimate_cv(Y, W)