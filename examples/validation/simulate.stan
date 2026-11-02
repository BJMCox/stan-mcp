data {
  int<lower=1> N;
}
generated quantities {
  real theta_sim = beta_rng(2, 2);
  int y_sim = binomial_rng(N, theta_sim);
  real joint_sim = beta_lpdf(theta_sim | 2, 2) + binomial_lpmf(y_sim | N, theta_sim);
}
