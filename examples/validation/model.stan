data {
  int<lower=1> N;
  int<lower=0, upper=N> y;
}
parameters {
  real<lower=0, upper=1> theta;
}
model {
  theta ~ beta(2, 2);
  y ~ binomial(N, theta);
}
generated quantities {
  int y_rep = binomial_rng(N, theta);
  real joint = beta_lpdf(theta | 2, 2) + binomial_lpmf(y | N, theta);
}
