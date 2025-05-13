#################################
######### Load packages #########
#################################

libs <- c("runjags", "coda", "MASS", "rstan")
needed <- libs[!vapply(libs, requireNamespace, logical(1), quietly = TRUE)]
if (length(needed)) install.packages(needed, repos = "https://cloud.r-project.org")

lapply(libs, library, character.only = TRUE)

if ("rstan" %in% libs) {
  options(mc.cores = 1L)
  rstan_options(auto_write = TRUE)
}

########################################
############ Sample dataset ############
########################################

rbinlogist <- function(n, m, mu, sigma_logit, nu) {
  plogis(rt(n, nu)*sigma_logit + mu) |>
    rbinom(n, m, prob = _)
}

parms <- list(
  N = 50,
  N_time = 10,
  m = 30,
  beta_0 = 1.5,
  beta_time = -0.05,
  beta_xt = 0.3,
  beta_cnt = 0.02,
  omega = 0.5,
  nu = 5,
  sigma_a_0 = 0.4,
  sigma_a_1 = 0.3,
  rho_a_01 = -0.5
)

Sigma <- matrix(c(parms$sigma_a_0^2,
                  parms$rho_a_01*parms$sigma_a_0*parms$sigma_a_1,
                  parms$rho_a_01*parms$sigma_a_0*parms$sigma_a_1,
                  parms$sigma_a_1^2), 2)

rand_eff <- MASS::mvrnorm(parms$N, mu = c(0, 0), Sigma = Sigma)
group <- sample(0:1, parms$N, TRUE)
cnt_cov <- rnorm(parms$N, 60, 5)
times <- 0:parms$N_time

simdat <- do.call(rbind, lapply(seq_len(parms$N), \(i) {
  eta <- parms$beta_0 +
    rand_eff[i, 1] +
    (rand_eff[i, 2] + parms$beta_time + parms$beta_xt*group[i])*times +
    parms$beta_cnt*cnt_cov[i]
  data.frame(
    id = i,
    time = times,
    group = group[i],
    cnt = cnt_cov[i],
    y = rbinlogist(length(times), parms$m, eta, parms$omega, parms$nu)
  )
}))

###################################################################
####################### JAGS implementation #######################
###################################################################

model_string <- "
  model {
    for (j in 1:2) {
      Psi[j, j] ~ dgamma(0.5, pow(50, -2))
    }
    Psi[1,2] <- 0; Psi[2,1] <- 0
    inv_Sigma ~ dwish(2*2*Psi, 3)
    for (k in 1:Nsubj) {
      a[k,1:2] ~ dmnorm(zero[], inv_Sigma[,])
    }
  
    beta_0 ~ dnorm(0, 0.001)
    beta_t ~ dnorm(0, 0.001)
    beta_xt ~ dnorm(0, 0.001)
    beta_cnt ~ dnorm(0, 0.001)
  
    sigma ~ dt(0, 0.25, 2) T(0,)
    inv_omega2 <- 1/pow(sigma, 2)
  
    lambda ~ dexp(0.75)
    nu ~ dgamma(2, lambda)
    log_nu <- log(nu)
  
    for (i in 1:Nobs) {
      eta[i] <- beta_0 +
                a[id[i],1] +
                (a[id[i],2] + beta_t + beta_xt*grp[i])*tm[i] +
                beta_cnt*cnt[i]
      tau[i] ~ dgamma(nu/2, nu/2)
      xi[i] ~ dnorm(eta[i], inv_omega2*tau[i])
      logit(p[i]) <- xi[i]
      y[i] ~ dbin(p[i], m[i])
    }
  }
"

jags_data <- with(simdat, list(
  Nsubj = max(id),
  Nobs = nrow(simdat),
  id = id,
  tm = time,
  grp = group,
  cnt = cnt,
  y = y,
  m = rep(parms$m, nrow(simdat)),
  zero = c(0, 0)
))

inits <- list(list(
  beta_0 = 0,
  beta_t = 0,
  beta_xt = 0,
  beta_cnt = 0,
  sigma = 1,
  nu = 10,
  ".RNG.name" = "base::Mersenne-Twister",
  ".RNG.seed" = 1234
))

monitor <- c(
  "beta_0", "beta_t", "beta_xt", "beta_cnt",
  "sigma", "nu", "log_nu",
  "inv_Sigma[1,1]", "inv_Sigma[1,2]",
  "inv_Sigma[2,1]", "inv_Sigma[2,2]"
)

fit <- run.jags(
  model = model_string,
  data = jags_data,
  inits = inits,
  monitor = monitor,
  n.chains = 1,
  burnin = 2000,
  sample = 4000,
  thin = 4,
  method = "rjags",
  modules = "glm",
  factories = "bugs::MNormal sampler off"
)

print(fit, quiet = TRUE)

###################################################################
####################### Stan implementation #######################
###################################################################

stan_code <- "
  data {
    int<lower=1> Nobs;
    int<lower=1> Nsubj;
    int<lower=1,upper=Nsubj> id[Nobs];
    vector[Nobs] tm;
    int<lower=0,upper=1> grp[Nobs];
    vector[Nobs] cnt;
    int<lower=0> y[Nobs];
    int<lower=1> m[Nobs];
  }
  
  parameters {
    real beta_0;
    real beta_t;
    real beta_xt;
    real beta_cnt;
  
    vector[Nsubj] a0;
    vector[Nsubj] a1;
    real<lower=0> sigma_a0;
    real<lower=0> sigma_a1;
    real<lower=-1,upper=1> rho;
  
    real<lower=0> sigma;
    real<lower=0> lambda;
    real<lower=2> nu;
  
    vector[Nobs] xi;
  }
  
  transformed parameters {
    matrix[2,2] Sigma;
    Sigma[1,1] = square(sigma_a0);
    Sigma[2,2] = square(sigma_a1);
    Sigma[1,2] = rho*sigma_a0*sigma_a1;
    Sigma[2,1] = Sigma[1,2];
  
    vector[Nobs] eta;
    for (i in 1:Nobs)
      eta[i] = beta_0 +
               a0[id[i]] +
               (a1[id[i]] + beta_t + beta_xt*grp[i])*tm[i] +
               beta_cnt*cnt[i];
  }
  
  model {
    beta_0 ~ normal(0, 1000);
    beta_t ~ normal(0, 1000);
    beta_xt ~ normal(0, 1000);
    beta_cnt ~ normal(0, 1000);
  
    sigma_a0 ~ student_t(2, 0, 50);
    sigma_a1 ~ student_t(2, 0, 50);
    rho ~ uniform(-1, 1);
  
    sigma ~ student_t(2, 0, 2);
  
    lambda ~ exponential(0.75);
    nu ~ gamma(2, lambda);
  
    for (k in 1:Nsubj) {
      vector[2] a_k = [ a0[k], a1[k] ]';
      a_k ~ multi_normal(rep_vector(0, 2), Sigma);
    }
  
    xi ~ student_t(nu, eta, sigma);
    y ~ binomial_logit(m, xi);
  }
  
  generated quantities {
    real log_lik[Nobs];
    for (i in 1:Nobs)
      log_lik[i] = binomial_logit_lpmf(y[i] | m[i], xi[i]);
  }
"

writeLines(stan_code, "blt_stan_model.stan")

stan_data <- with(simdat, list(
  Nobs = nrow(simdat),
  Nsubj = max(id),
  id = id,
  tm = time,
  grp = group,
  cnt = cnt,
  y = y,
  m = rep(parms$m, nrow(simdat))
))

fit_stan <- stan(
  file = "blt_stan_model.stan",
  data = stan_data,
  chains = 1,
  iter = 2000,
  seed = 12345,
  control = list(adapt_delta = 0.9)
)

print(fit_stan,
      pars = c("beta_0", "beta_t", "beta_xt", "beta_cnt",
               "sigma", "nu", "sigma_a0", "sigma_a1", "rho"),
      probs = c(0.025, 0.5, 0.975))