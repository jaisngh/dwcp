# loading libraries
library(cmdstanr)
options(cmdstanr_warn_inits = FALSE)

# reading in model
# model <- cmdstan_model("estimator/dwcp.stan", force_recompile = TRUE)
cv_model <- cmdstan_model("estimator/dwcp_cv.stan", force_recompile = TRUE)


# function to estimate hyperparameters via cross-validation
estimate_cv <- function(Y, W, num_iter=20, num_folds=5) {
  hyperparameters <- estimate_cv_test(Y, W, num_iter, num_folds)
  lambda <- hyperparameters$lambda
  tau <- hyperparameters$tau
  
  data <- list(N = N, T = T, Y = Y, W = W, K = num_folds, lambda = lambda, tau = tau)

  fit <- model$optimize(
    data = data,
    init_alpha = 0.00001,
    algorithm = "lbfgs" 
  )
  estimates <- fit$summary()

  # getting predicted Y(0)
  Y_hat <- matrix(NA, N, T)
  for (n in 1:N) {
    for (t in 1:T) {
      Y_hat[n, t] <- estimates %>% filter(grepl(paste0("Y_hat[", n, ",", t, "]"), variable)) %>% pull(estimate)
    }
  }

  return(list(Y_hat = Y_hat))
}


# function to compute max lambda for given data
# kinda convoluted see Athey github for reference
estimate_max_lambda <- function(Y, W) { 
  N <- nrow(Y)
  T <- ncol(Y)
  mask <- matrix(1, N, T) - W

  # TWFE for observed data (matrix completion with all L = 0 or infinite penalty)
  Y_std <- (Y - mean(Y)) / sd(Y)
  alpha_std <- rowSums(Y_std * mask) / rowSums(mask)
  gamma_std <- colSums(Y_std * mask) / colSums(mask)

  E <- matrix(rep(alpha_std, T), N, T) + matrix(rep(gamma_std, each=N), N, T)
  P_omega <- (Y_std - E) * mask
  max_lambda <- 2 * max(svd(P_omega)$d) 
  return(max_lambda)
}


estimate_cv_test <- function(Y, W, num_iter, num_folds) {
  N <- nrow(Y)
  T <- ncol(Y)
  data <- list(N = N, T = T, Y = Y, W = W, K = num_folds)

  # grid for tau
  tau_grid <- seq(0, 1, length.out=10)

  # finding max lambda
  max_lambda <- estimate_max_lambda(Y, W)
  print(paste0("Max lambda: ", max_lambda))

  # generating linearly-spaced (in log10 space) descending grid for lambda
  lambda_grid_argument <- seq(log10(max_lambda), log10(max_lambda) - 3, length.out=num_iter-1)
  lambda_grid <- 10^lambda_grid_argument
  lambda_grid <- c(lambda_grid, 0)

  # setting up hyperparameter tracking
  count <- 1
  best_rmse <- Inf
  best_lambda <- NA
  best_tau <- NA

  # grid searching over lambda
  for (tau in tau_grid) {
    for (lambda in lambda_grid) { 
      data$lambda <- lambda
      data$tau <- tau 
      fit <- cv_model$optimize(data = data, algorithm = "lbfgs", show_messages = FALSE)
      estimates <- fit$summary()
      rmse <- estimates %>% filter(grepl("RMSE", variable)) %>% pull(estimate)
      if (rmse < best_rmse) {
        best_rmse <- rmse
        best_lambda <- lambda
        best_tau <- tau
      }
      print(paste0("Iteration ", count, ":: lambda=", lambda, " tau=", tau, " RMSE=", rmse))
      count <- count + 1
    }
  }
  return(list(lambda=best_lambda, tau=best_tau))
}
