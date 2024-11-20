library(tidyverse)
library(ggplot2)
library(cmdstanr)
library(ggthemes)
theme_set(theme_few())

# loading estimator script
source("estimator/estimate.r")

# loading smoke outcome data
smoke_outcome <- read_csv("data/smoke/smok_outcome.csv", col_names = FALSE, show_col_types = FALSE)
smoke_treatment <- read_csv("data/smoke/smok_treatment.csv", col_names = FALSE, show_col_types = FALSE)
colnames(smoke_outcome) <- 1:ncol(smoke_outcome)
colnames(smoke_treatment) <- 1:ncol(smoke_treatment)

# running CV
Y <- as.matrix(smoke_outcome)
W <- as.matrix(smoke_treatment)
estimate_cv(Y, W)