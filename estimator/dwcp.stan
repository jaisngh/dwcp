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
}

transformed data {
  /////////////////////////////
  // Standardizing Variables //
  /////////////////////////////
  real Y_mean = mean(Y);
  real Y_standard_deviation = sd(Y);
  matrix[N, T] Y_std = (Y - Y_mean) / Y_standard_deviation;

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


  // converting ||L||* penalty to standard deviation
  real sigma_L;
  if (lambda > 0) {
    sigma_L = 1/sqrt(lambda);
  } else {
    sigma_L = positive_infinity();
  }
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
}

transformed parameters {
  // generating predicted outcome
  vector[O] Y_std_hat;
  for (i in 1:O) {
    int n = observed_indices[i, 1];
    int t = observed_indices[i, 2];
    Y_std_hat[i] = alpha_std[n] + gamma_std[t] + L_std[n, t];
  }
}

model {
  // // /////////////////////////////////////////
  // // // Estimating Propensity Score Weights //
  // // /////////////////////////////////////////
  // if (sigma_E > 0) {
  //   // prior on nuclear norm of propensity score
  //   real nuclear_norm_propensity_score = sum(singular_values(propensity_matrix));
  //   nuclear_norm_propensity_score ~ normal(0, sigma_E);
  // }
  // // modelling treatment status as a function of propensity scores and observed Y values
  // to_vector(W) ~ normal(to_vector(propensity_hat), 1);


  // setting prior on L_std if sigma_L < infinity 
  if (sigma_L < positive_infinity()) {
    real nuclear_norm_L_std = sqrt(sum(singular_values(L_std)));
    nuclear_norm_L_std ~ normal(0, sigma_L);
  }


  // adding data to likelihood
  {
    vector[O] ll;
    for (i in 1:O) {
      int n = observed_indices[i, 1];
      int t = observed_indices[i, 2];
      ll[i] = normal_lpdf(Y_std_observed[i] | Y_std_hat[i], 1);
    }
    target += sum(ll);
  }
  
}   

generated quantities {
  // generating unstandardized parameters and predictions
  matrix[N, T] Y_hat;
  for (n in 1:N) {
    for (t in 1:T) {
      Y_hat[n, t] = Y_standard_deviation * (alpha_std[n] + gamma_std[t] + L_std[n, t]) + Y_mean;
    }
  }
}

