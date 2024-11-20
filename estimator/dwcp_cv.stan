functions {
  // function to create held-out training treatment matrix
  matrix create_training_matrix_rng(matrix W, int K) {
    // if we want to get full estimates (no cv)
    if (K == 1) {
      return W;
    }
    int N = rows(W);
    int T = cols(W);
    matrix[N, T] W_train = rep_matrix(1, N, T);
    for (n in 1:N) {
      for (t in 1:T) {
        if (W[n, t] == 0) {
          W_train[n, t] = bernoulli_rng(0.2);
        }
      }
    }
    return W_train;
  }
}

data {
  /////////////////
  // Data Inputs //
  /////////////////
  // number of units
  int N; 
  // number of time periods
  int T; 
  // N x T matrix of outcome variable
  matrix[N, T] Y; 
  // N x T matrix of treatment variable (should be 0 or 1)
  matrix[N, T] W;

  /////////////////////
  // Hyperparameters //
  /////////////////////
  // penalty on ||L||* 
  real<lower=0> lambda;
  // softmax temperature
  real<lower=0> tau;
  // number of folds
  int<lower=1> K;
}

transformed data {
  /////////////////////////////
  // Creating K-Folds for CV //
  /////////////////////////////
  // folds
  array[K] matrix[N, T] W_train;
  for (k in 1:K) {
    W_train[k] = create_training_matrix_rng(W, K);
  }

  ////////////////////////////////////
  // Standardizing outcome variable //
  ////////////////////////////////////
  real Y_mean = mean(Y);
  real Y_standard_deviation = sd(Y);
  matrix[N, T] Y_std = (Y - Y_mean) / Y_standard_deviation;

  ////////////////////////////////////////
  // Flattening Data with Observed Y(0) //
  ////////////////////////////////////////
  // number of units with observed Y(0) in each fold
  array[K] int O;
  for (k in 1:K) {
    // number of units with observed Y(0)
    O[k] = to_int(N*T - sum(W_train[k]));
    // indices of observed Y(0)
  }
  
  int max_O = max(O);

  array[K, max_O, 2] int observed_indices;
  array[K] vector[max_O] Y_std_observed;

  for (k in 1:K) {
    int o = 1;
    for (n in 1:N) {
      for (t in 1:T) {
        if (W_train[k, n, t] == 0) {
          observed_indices[k, o, 1] = n;
          observed_indices[k, o, 2] = t;
          Y_std_observed[k, o] = Y_std[n, t];
          o += 1;
        }
      }
    }
  }
  ////////////////////////////////////////
  // Lasso Penalty Parameters //////////
  ////////////////////////////////////////
  // converting ||L||* penalty to standard deviation
  real sigma_L;
  if (lambda > 0) {
    sigma_L = 1/sqrt(lambda);
  } else {
    sigma_L = positive_infinity();
  }


  ////////////////////////////////////////
  // Propensity Score Data //////////////
  ////////////////////////////////////////
  // mean treatment time for each fold
  vector[K] mu_T;
  for (k in 1:K) {
    mu_T[k] = 0;
    for (n in 1:N) {
      for (t in 1:T) {
        if (W_train[k, n, t] == 1) {
          mu_T[k] += t;
        }
      }
    }
    mu_T[k] = mu_T[k] / sum(W_train[k]);
  }

  // time matrix
  vector[T] time_vector;
  for (t in 1:T) {
    time_vector[t] = t;
  }
  matrix[N, T] time_matrix = rep_matrix(time_vector, N)';

  // integer array of flattened W_train
  array[K, N*T] int W_prop_array;
  for (k in 1:K) {
    for (n in 1:N) {
      for (t in 1:T) {
        W_prop_array[k, n + (t-1)*N] = to_int(W_train[k, n, t]);
      }
    }
  }
}

parameters {
  ////////////////////////////////////////
  /// Creating standardized parameters ///
  ////////////////////////////////////////
  // unit fixed-effects
  array[K] vector[N] alpha_std;
  // time fixed-effects
  array[K] vector[T] gamma_std;
  // Factor matrix
  array[K] matrix[N, T] L_std;

  //////////////////////////////////
  // Propensity Score Parameters //
  /////////////////////////////////
  // intercept term
  vector[K] prop_intercept;
  // Y coefficient
  vector[K] prop_Y;
  // time coefficient
  vector[K] prop_time;
}

transformed parameters {
  // generating logit argument for propensity score
  array[K] matrix[N, T] prop_logit;
  for (k in 1:K) {
    prop_logit[k] = prop_intercept[k] + prop_Y[k] * Y_std + prop_time[k] * (time_matrix - mu_T[k]);
  }


  // generating predicted outcome
  array[K] vector[max_O] Y_std_hat;
  for (k in 1:K) {
    for (i in 1:O[k]) {
      int n = observed_indices[k, i, 1];
      int t = observed_indices[k, i, 2];
      Y_std_hat[k, i] = alpha_std[k, n] + gamma_std[k, t] + L_std[k, n, t];
    }
  }
}

model {
  /////////////////////////////////////////
  // Estimating Propensity Score Weights //
  /////////////////////////////////////////
  for (k in 1:K) {
    W_prop_array[k] ~ bernoulli_logit(to_vector(prop_logit[k]));
  }


  // setting prior on L_std if sigma_L < infinity 
  if (sigma_L < positive_infinity()) {
    for (k in 1:K) {
      real nuclear_norm_L_std = sqrt(sum(singular_values(L_std[k])));
      nuclear_norm_L_std ~ normal(0, sigma_L);
    }
  }

  // constructing weights
  array[K] vector[max_O] weights;
  for (k in 1:K) {
    weights[k] = rep_vector(0, max_O);
    for (i in 1:O[k]) {
      int n = observed_indices[k, i, 1];
      int t = observed_indices[k, i, 2];
      weights[k, i] = exp(tau * prop_logit[k, n, t]);
    }
    weights[k] = O[k] * (weights[k] / sum(weights[k]));
  }

  // adding data to likelihood
  {
    for (k in 1:K) {
      vector[O[k]] ll;
      // getting weights based on propensity scores
      for (i in 1:O[k]) { 
        ll[i] = weights[k, i] * normal_lpdf(Y_std_observed[k, i] | Y_std_hat[k, i], 1);
      }
      target += sum(ll);
    }
  }
  
}   

generated quantities {
  // calculating RMSE across folds
  matrix[N, T] W_mirror = rep_matrix(1, N, T) - W;
  real RMSE = 0;
  for (k in 1:K) {
    matrix[N, T] Y_hat_temp = rep_matrix(alpha_std[k], T) + rep_matrix(gamma_std[k], N)' + L_std[k];
    matrix[N, T] validation_mask = W_mirror .* W_train[k];
    real sum_k = sum(((Y_std - Y_hat_temp) .* validation_mask).^2) / sum(validation_mask);
    RMSE += sqrt(sum_k);
  }

  RMSE = RMSE / K;

  // final output
  matrix[N, T] Y_hat;
  Y_hat = Y_mean + Y_standard_deviation * (rep_matrix(alpha_std[1], T) + 
                                           rep_matrix(gamma_std[1], N)' + 
                                           L_std[1]);
}

