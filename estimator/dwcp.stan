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
  // standard deviation of nuclear norm prior
  real<lower=0> sigma_L;
  // hyperparameter for unit and time distance weights
  real<lower=0> lambda_N;
  real<lower=0> lambda_T;
}

transformed data {
  /////////////////////////////
  // Standardizing Variables //
  /////////////////////////////
  real Y_mean = mean(Y);
  real Y_standard_deviation = sd(Y);
  matrix[N, T] Y_std = (Y - Y_mean) ./ Y_standard_deviation;

  ////////////////////////////////////////
  // Flattening Data with Observed Y(0) //
  ////////////////////////////////////////
  // number of units with observed Y(0)
  int O = to_int(N*T - sum(W));
  // indices of observed Y(0)
  array[O, 2] int observed_indices;
  // observed Y(0)
  vector[O] Y_std_observed;
  int o = 1;
  for (n in 1:N) {
    for (t in 1:T) {
      if (W[n, t] == 0) {
        observed_indices[o, 1] = n;
        observed_indices[o, 2] = t;
        Y_std_observed[o] = Y_std[n, t];
        o += 1;
      }
    }
  } 


  //////////////////////////////////
  // Time and Unit Weights (DWCP) //
  /////////////////////////////////
  // getting adoption times for each unit
  // marking treated units
  vector[N] adoption_times = rep_vector(2 * T, N);
  vector[N] treated_units = rep_vector(0, N);
  for (n in 1:N) {
    int counter = 1;
    for (t in 1:T) {
      if (W[n, t] == 1) {
        treated_units[n] = 1;
        adoption_times[n] = counter;
        break;
      }
      counter += 1;
    }
  }

  // getting time and unit weights
  vector[O] time_distance;
  vector[O] unit_distance;
  vector[O] dwcp_weights;

  for (i in 1:O) {
    int n = observed_indices[i, 1];
    int t = observed_indices[i, 2];
    // time-distance is the shortest time to nearest adoption-time across all units
    time_distance[i] = norm2((adoption_times - t) .* treated_units);

    // calculating unit-distance
    // calculating ||Y_n - Y_j||^2 for all j in [N]
    matrix[N, T] unit_shift_matrix = (Y_std -  rep_matrix(Y_std[n, : ], N)) ;
    vector[N] unit_distance_vector = (unit_shift_matrix .* unit_shift_matrix) * rep_vector(1, T);
    // // calculating mean distance from non-self treated units 
    // real unit_distance_total = 0;
    // int unit_distance_count = 0;
    // for (unit in 1:N) {
    //   if (sum(W[unit, : ]) > 0 && unit != n) {
    //     unit_distance_total += unit_distance_vector[unit];
    //     unit_distance_count += 1;
    //   }
    // }
    unit_distance[i] = norm2(unit_distance_vector .* treated_units);

    // calculating DWCP weights
    dwcp_weights[i] = exp(-(lambda_T * time_distance[i] + lambda_N * unit_distance[i]));
  }

  // normalizing weights
  dwcp_weights = (dwcp_weights / sum(dwcp_weights)) * O;
}

parameters {
  ////////////////////////////////////////
  /// Creating standardized parameters ///
  ////////////////////////////////////////
  // unit fixed-effects
  vector[N] alpha_std;
  // time fixed-effects
  vector[T] gamma_std;
  // Factor matrix
  matrix[N, T] L_std;


  //////////////////////////////
  // Propensity Score Weights //
  /////////////////////////////
  // // propensity score weights
  // matrix[N, T] propensity_score;
}

transformed parameters {
  // generating predicted outcome
  vector[O] Y_std_hat;
  for (i in 1:O) {
    int n = observed_indices[i, 1];
    int t = observed_indices[i, 2];
    if (sigma_L > 0) {
      Y_std_hat[i] = alpha_std[n] + gamma_std[t] + L_std[n, t];
    } else {
      Y_std_hat[i] = alpha_std[n] + gamma_std[t];
    }
  }
}

model {
  // /////////////////////////////////////////
  // // Estimating Propensity Score Weights //
  // /////////////////////////////////////////
  // // prior on nuclear norm of propensity score
  // real nuclear_norm_propensity_score = sum(singular_values(propensity_score));
  // nuclear_norm_propensity_score ~ normal(0, sigma_L);
  // // modelling treatment status as a function of propensity scores
  // to_vector(W) ~ normal(to_vector(propensity_score), 1);


  // setting prior on L_std
  if (sigma_L > 0) {
    real nuclear_norm_L_std = sum(singular_values(L_std));
    nuclear_norm_L_std ~ normal(0, sigma_L);
  }

  // adding data to likelihood
  {
    for (i in 1:O) {
      int n = observed_indices[i, 1];
      int t = observed_indices[i, 2];
      real weight = dwcp_weights[i];
      target += weight * normal_lpdf(Y_std_observed[i] | Y_std_hat[i], 1);
    }
  }
}   

generated quantities {
  // generating unstandardized parameters and predictions
  vector[N] alpha = alpha_std * Y_standard_deviation;
  vector[T] gamma = gamma_std * Y_standard_deviation;
  matrix[N, T] Y_hat = rep_matrix(alpha, T) + rep_matrix(gamma, N)' + L_std + Y_mean;

  // average treatment effect
  real ate = sum((Y - Y_hat) .* W) / sum(W);

  // weights for debug
  vector[O] dwcp_weights_debug = dwcp_weights;
  vector[O] time_distance_debug = time_distance;
  vector[O] unit_distance_debug = unit_distance;
}

