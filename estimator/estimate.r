# loading cmdstanr library
library(cmdstanr)
options(cmdstanr_warn_inits = FALSE)
model <- cmdstan_model("estimator/dwcp.stan", force_recompile = TRUE)


# function to estimate hyperparameters via cross-validation
estimate_cv <- function(Y, W, num_folds=5, num_iters=15) {
  N <- nrow(Y)
  T <- ncol(Y)
  data_folds <- create_folds(N, T, W, num_folds)

  # grid searching over hyperparameters
  total_rmse <- 0
  count <- 1
  best_rmse <- Inf
  best_lambda_N <- NA
  best_lambda_T <- NA
  best_sigma_L <- NA

  for (lambda_T in seq(0, 5, length.out = num_iters)) {
    for (lambda_N in seq(0, 5, length.out = num_iters)) {
      # initializing list of L_init values
      L_init_list <- list()
      unitFE_init_list <- list()
      timeFE_init_list <- list()
      for (k in 1:num_folds) {
        L_init_list[[k]] <- matrix(0, N, T)
        unitFE_init_list[[k]] <- rep(0, N)
        timeFE_init_list[[k]] <- rep(0, T)
      }

      for (sigma_L in seq(0, 10, length.out = num_iters)) {
        for (k in 1:num_folds) {
          data <- data_folds[[k]]
          data$sigma_L <- sigma_L
          data$lambda_N <- lambda_N
          data$lambda_T <- lambda_T

          # estimating
          estimates <- estimate_dwcp(data, L_init_list[[k]], unitFE_init_list[[k]], timeFE_init_list[[k]])
          Y_hat <- estimates$Y_hat
          L_init_list[[k]] <- estimates$L_std
          unitFE_init_list[[k]] <- estimates$unitFE
          timeFE_init_list[[k]] <- estimates$timeFE

          # calculating RMSE on validation data
          W <- data$W
          W_true <- data$W_true
          validation <- (data$Y - Y_hat) * (as.vector(W) == 1 & as.vector(W_true) == 0)
          rmse <- sqrt(mean(validation^2))
          total_rmse <- total_rmse + rmse
        }
        avg_rmse <- total_rmse / num_folds
        if (avg_rmse < best_rmse) {
          best_rmse <- avg_rmse
          best_lambda_N <- lambda_N
          best_lambda_T <- lambda_T
          best_sigma_L <- sigma_L
        }
        # printing progress
        if (count %% num_iters == 0) {
          print(paste0("Best Sigma_L and RMSE for lambda_N: ", lambda_N, " and lambda_T: ", lambda_T))
          print(paste0("sigma_L: ", best_sigma_L, " RMSE: ", best_rmse))
        }
        count <- count + 1
      }
    }
  }

  # fitting final model
  data <- list(
    N = N,
    T = T,
    Y = Y,
    W = W,
    sigma_L = best_sigma_L,
    lambda_N = best_lambda_N,
    lambda_T = best_lambda_T
  )

  fit <- model$optimize(data = data, algorithm = "lbfgs", show_messages = TRUE)
  parameters <- fit$summary()
  return(list(data=data, parameters=parameters))
}

create_folds <- function(N, T, W, num_folds) {
  # choosing subset of Y(0) observed points
  sample_observation_size <- round((sum(W)^2)/(N*T))
  training_indices <- sample(which(W == 0), sample_observation_size, replace = FALSE)
  # masking W matrix for validation setup
  W_validation <- matrix(1, N, T)
  W_validation[training_indices] <- 0
  data_folds <- list()
  for (k in 1:num_folds) {
    data_folds[[k]] <- list(
      N = N,
      T = T,
      Y = Y,
      W = W_validation,
      W_true = W
    )
  }
  return(data_folds)
}


# function to estimate model given data and initial L_std point
# returns L_std and Y_hat matrices
estimate_dwcp <- function(data, L_init, unitFE_init, timeFE_init) {
  N <- data$N
  T <- data$T

  fit <- model$optimize(
    data = data,
    init_alpha = 0.001,
    algorithm = "lbfgs",
    show_messages = FALSE,
    init = function() list(L_std = L_init, alpha_std = unitFE_init, gamma_std = timeFE_init)
  )
  estimates <- fit$summary()

  # getting estimates
  L_std <- matrix(estimates %>% filter(grepl("L_std", variable)) %>% pull(estimate), N, T) 
  unitFE <- estimates %>% filter(grepl("alpha_std", variable)) %>% pull(estimate)
  timeFE <- estimates %>% filter(grepl("gamma_std", variable)) %>% pull(estimate)
  Y_hat <- matrix(estimates %>% filter(grepl("Y_hat", variable)) %>% pull(estimate), N, T)

  return(list(L_std = L_std, unitFE = unitFE, timeFE = timeFE, Y_hat = Y_hat))
}