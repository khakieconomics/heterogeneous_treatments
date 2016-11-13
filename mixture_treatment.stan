data {
  int N; // observations
  int P; // covariates
  vector<lower = 0, upper = 1>[N] treatment; // binary treatment
  matrix[N, P] X; // covariates
  vector[N] Y;
}
transformed data {
  matrix[N, P+1] X2;
  X2 = append_col(rep_vector(1.0, N), X);
}
parameters {
  real alpha; // intercept
  vector[P] beta; // regression coefficients
  real<upper = 0> tau_1; // negative treatment effect
  real<lower = 0> tau_3; // positive treatment effect
  matrix[2, P+1] gamma_raw; // treatment effect model 
  real<lower = 0> sigma;
}
transformed parameters {
  matrix[N, 3] theta;
  {
  matrix[3, P+1] gamma; // treatment effect model parameters (zero centered)
  gamma = append_row(gamma_raw, rep_row_vector(0.0, P+1));  
  for(n in 1:N) {
    theta[n] = softmax(gamma*X2[n]')';
  }
  }
}
model {
  alpha ~ normal(0, 1);
  beta ~ normal(0, 1);
  tau_1 ~ normal(0, 1);
  tau_3 ~ normal(0, 1);
  to_vector(gamma_raw) ~ normal(0, 1);
  sigma ~ cauchy(0, 1);
  
  for(n in 1:N) {
    vector[3] temp;
    
    temp[1] = log(theta[n,1]) + normal_lpdf(Y[n] | alpha + X[n]*beta + tau_1*treatment[n], sigma);
    temp[2] = log(theta[n,2]) + normal_lpdf(Y[n] | alpha + X[n]*beta, sigma);
    temp[3] = log(theta[n,3]) + normal_lpdf(Y[n] | alpha + X[n]*beta + tau_3*treatment[n], sigma);
    
    target += log_sum_exp(temp);
  }
}
generated quantities {
  vector[N] treatment_effect;
  vector[3] tau;
  tau[1] = tau_1;
  tau[2] = 0.0;
  tau[3] = tau_3;
  
  for(n in 1:N) {
    treatment_effect[n] = theta[n]*tau;
  }
}