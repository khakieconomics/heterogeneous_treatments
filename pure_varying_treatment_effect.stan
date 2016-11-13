data {
  int N;
  int P;
  vector<lower = 0, upper = 1>[N] treatment;
  vector[N] Y;
  matrix[N, P] X;
}
transformed data {
  matrix[N, P+1] X1;
  X1 = append_col(rep_vector(1.0, N), X);
}
parameters {
  vector[P+1] gamma;
  vector[P+1] beta;
  vector<lower = 0>[2] scale;
  vector[N] tau;
}
model {
  gamma ~ normal(0, 1);
  beta ~ normal(0, 1);
  scale ~ cauchy(0, 1);
  {
    matrix[N, 2] depvars;
    matrix[N, 2] cond_means;
    depvars = append_col(Y, tau);
    cond_means = append_col(X1*beta + tau .* treatment, X1*gamma);
    for(n in 1:N) {
      depvars[n] ~ normal(cond_means[n], scale);
    }
  }
}