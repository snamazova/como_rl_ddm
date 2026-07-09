# Validation script: generate reference RLDDM output from the original R code.
#
# Run with:
#   Rscript validate_rlddm.R
#
# It sources the original course file, simulates one synthetic participant,
# and saves the task environment + simulated data + rlddm_run fit so that
# validate_rlddm.py can compare the Python port against it.

source("../Full_R/cognitive_modelling_course_RLDDM_full.r")

set.seed(42)

task_environment <- simulate_task_environment(
  n_trials = 140,
  means = c(1.0, 0.0),
  sds = c(1.0, 1.0)
)

pars <- c(
  alpha = 0.25,
  v_intercept = 0.0,
  v_scale = 1.0,
  a = 3.0,
  w = 0.5,
  t0 = 0.25
)

# Simulate synthetic data from the model
sim <- rlddm_simulate(
  task_environment = task_environment,
  pars = pars,
  initial_values = c(0, 0)
)

# Compute the log-likelihood of the simulated data under the same parameters
fit <- rlddm_run(
  data = sim$data,
  pars = pars,
  initial_values = c(0, 0)
)

# Save outputs for Python comparison
write.csv(task_environment, "validation_task_environment.csv", row.names = FALSE)
write.csv(sim$data, "validation_simulated_data.csv", row.names = FALSE)
write.csv(sim$values, "validation_values.csv", row.names = FALSE)
write.csv(data.frame(drifts = sim$drifts), "validation_drifts.csv", row.names = FALSE)
write.csv(data.frame(prediction_errors = sim$prediction_errors), "validation_prediction_errors.csv", row.names = FALSE)
write.csv(data.frame(log_lik = fit$log_lik), "validation_log_lik.csv", row.names = FALSE)
cat("summed_log_lik:", fit$summed_log_lik, "\n")

# Save the parameters too
write.csv(data.frame(t(pars)), "validation_pars.csv", row.names = FALSE)
