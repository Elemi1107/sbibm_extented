import logging
import numpy as np
import torch
from typing import Optional
from torch.distributions import MultivariateNormal
from pyvbmc import VBMC
from sbibm.tasks.task import Task

logging.getLogger().setLevel(logging.INFO)


def compute_synthetic_likelihood(thetas, simulator, simulations_per_eval, obs, diag_eps=0.01, num_bootstrap=1000):
    """Compute synthetic likelihood using simulations

        Args:
            thetas: Parameters
            simulations_per_eval: Number of simulations used per evaluation
            obs: Observation or summary statistics
            diag_eps: A small constant to keep covariance matrix positive infinite
            num_boostrap: Number of bootstrap samplings

        Returns:
            log-likelihood and noise estimates for given parameters.
        """
    log = logging.getLogger(__name__)
    log_likelihoods = []
    noise_estimates = []

    for i in range(thetas.shape[0]):
        simulated_data = simulator(
            thetas[i, :].reshape(1, -1).repeat(simulations_per_eval, 1))  # (simulations_per_eval, feature_dim)


        mu_hat = torch.mean(simulated_data, dim=0)  # (feature_dim,)
        cov_hat = torch.cov(simulated_data.T) + diag_eps * torch.eye(mu_hat.shape[0])

        try:
            torch.linalg.cholesky(cov_hat)
        except torch._C._LinAlgError:
            log.warning("Covariance matrix is not positive definite. Returning -1e6 as log-likelihood.")
            log_likelihoods.append(torch.tensor(-1e6).item())
            noise_estimates.append(torch.tensor(1e6).item())
            continue

        dist = MultivariateNormal(mu_hat, covariance_matrix=cov_hat)
        log_likelihood = dist.log_prob(obs)
        log_likelihoods.append(log_likelihood.item())

        # estimate log likelihood noise using bootstrap resampling
        bootstrap_log_likelihoods = []
        bs_valild = True
        for _ in range(num_bootstrap):
            resampled_indices = torch.randint(0, simulated_data.shape[0], ( simulated_data.shape[0],))
            resampled_data = simulated_data[resampled_indices]

            mu_resampled = torch.mean(resampled_data, dim=0)
            cov_resampled = torch.cov(resampled_data.T) + diag_eps * torch.eye(mu_resampled.shape[0])
            try:
                torch.linalg.cholesky(cov_resampled)
            except torch._C._LinAlgError:
                log.warning("Covariance matrix for bootstrap resampling is not positive definite. Returning 1e6 as noise estimate")
                noise_estimates.append(torch.tensor(1e6).item())
                bs_valild = False
                break

            dist_resampled = MultivariateNormal(mu_resampled, covariance_matrix=cov_resampled)
            bootstrap_log_likelihoods.append(dist_resampled.log_prob(obs).item())

        if not bs_valild:
            continue

        noise_std = torch.std(torch.tensor(bootstrap_log_likelihoods))
        noise_estimates.append(noise_std.item())

    return torch.tensor(log_likelihoods), torch.tensor(noise_estimates)

def run(
    task: Task,
    num_samples: int,
    noisy_likelihood: bool = True,
    num_simulations: Optional[int] = None,
    simulations_per_eval: Optional[int] = None,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
) -> (torch.Tensor, int, Optional[torch.Tensor]):
    """Runs PyVBMC to perform variational Bayesian inference.

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_simulations: Number of total simulations (only valid to noisy likelihood)
        simulations_per_eval: Number of simulations used per evaluation
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`

    Returns:
        Samples from the variational posterior, number of simulations, and ELBO.
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)
    if (num_simulations is not None) == (simulations_per_eval is not None) and noisy_likelihood:
        raise ValueError("You must specify exactly one of `num_simulations` or `simulations_per_eval`.")


    log = logging.getLogger(__name__)
    log.info(f"Running PyVBMC with noisy log likelihood = {noisy_likelihood}")

    if num_observation is not None:
        observation = task.get_observation(num_observation)

    if task.name == 'svar' and not noisy_likelihood:
        task.set_raw(True)

    if not noisy_likelihood:
        log_prob_fn = task._get_log_prob_fn(num_observation=num_observation, posterior=True)
        options = None
    else:
        simulator = task.get_simulator()
        sim_evals_map = {
            10 ** 3: 20,
            10 ** 4: 100,
            10 ** 5: 400
        }
        if simulations_per_eval is None:
            evaluations = sim_evals_map.get(num_simulations)
            options = {
                "specify_target_noise": True,
                "max_fun_evals": evaluations,
                "min_fun_evals": evaluations,
                "min_iter": 0,
                "max_iter": 100000000,
            }
            simulations_per_eval = int(num_simulations / evaluations)
        else:
            # run until converge
            options = {
                "specify_target_noise": True,
            }

        def log_prob_fn(thetas):
            log_likelihoods, noise_std = compute_synthetic_likelihood(thetas, simulator, simulations_per_eval, observation)
            log_priors = task.prior_dist.log_prob(thetas)
            return (log_likelihoods + log_priors).numpy().flatten(), noise_std.numpy().flatten().item()

    # log_prob_fn = task._get_log_prob_fn(num_observation=num_observation, posterior=True)


    D = task.dim_parameters
    LB = np.full(D, -np.inf)
    UB = np.full(D, np.inf)


    prior_params = task.get_prior_params()
    prior_cls = str(task.prior_dist)
    # LogNormal prior
    if "LogNormal" in prior_cls:
        # original param space
        # SIR task
        prior_mean = np.exp(prior_params["loc"].numpy() + 0.5 * prior_params["scale"].numpy() ** 2)
        prior_var = (np.exp(prior_params["scale"].numpy() ** 2) - 1) * np.exp(
            2 * prior_params["loc"].numpy() + prior_params["scale"].numpy() ** 2)
        prior_cov = np.diag(prior_var)
        LB = np.full(D, 0)
        UB = np.full(D, 1)
        PLB = np.maximum(prior_mean - np.sqrt(np.diag(prior_cov)), 1e-6)  # PLB > 0
        PUB = prior_mean + np.sqrt(np.diag(prior_cov))
    elif "MultivariateNormal" in prior_cls:
        # GLM task
        prior_mean = prior_params["loc"].numpy()
        prior_cov = np.linalg.inv(prior_params["precision_matrix"].numpy())
        PLB = prior_mean - np.sqrt(np.diag(prior_cov))
        PUB = prior_mean + np.sqrt(np.diag(prior_cov))

    elif "Uniform" in prior_cls:
        LB = prior_params["low"].numpy()
        UB = prior_params["high"].numpy()
        PLB = prior_params["low"].numpy()
        PUB = prior_params["high"].numpy()
        prior_mean = (PLB + PUB)/2
        prior_mean +=  np.random.uniform(-0.1, 0.1, size=prior_mean.shape)


    else:
        prior_mean = prior_params["loc"].numpy()
        prior_cov = np.diag(prior_params["scale"].numpy() ** 2)
        # PLB / PUB
        PLB = prior_mean - np.sqrt(np.diag(prior_cov))
        PUB = prior_mean + np.sqrt(np.diag(prior_cov))

    # prior mean as initial point of VBMC
    x0 = prior_mean

    def log_prob_wrapper(theta):
        theta = np.atleast_2d(theta)  # ensure 2 dim
        if noisy_likelihood:
            return log_prob_fn(torch.tensor(theta, dtype=torch.float32))
        else:
            return log_prob_fn(torch.tensor(theta, dtype=torch.float32)).item()

    # run PyVBMC
    vbmc = VBMC(log_prob_wrapper, x0, LB, UB, PLB, PUB,options=options)

    vp, results = vbmc.optimize()


    # sample posterior
    posterior_samples,_ = vp.sample(num_samples)

    elbo = results["elbo"]


    import matplotlib.pyplot as plt

    active_sampling_times = [timer._durations.get("active_sampling", 0) for timer in vbmc.iteration_history["timer"]]


    # Print active sampling time for each iteration
    # for i, act_time in enumerate(active_sampling_times):
    #     print(f"Iteration {i}: Active sampling time = {act_time:.2f} seconds")


    # # Convert to NumPy arrays and ensure numerical values
    # elbo_trace = np.array(vbmc.iteration_history["elbo"], dtype=np.float64)
    # elbo_sd_trace = np.array(vbmc.iteration_history["elbo_sd"], dtype=np.float64)
    # print(elbo_trace)
    # print(elbo_sd_trace)
    # # Ensure there are no NaNs or None values
    # elbo_trace = np.nan_to_num(elbo_trace, nan=np.nan)  # Replace None with NaN
    # elbo_sd_trace = np.nan_to_num(elbo_sd_trace, nan=np.nan)
    #
    # iterations = np.arange(len(elbo_trace))
    #
    # # Plot ELBO trace
    # plt.figure(figsize=(8, 5))
    # plt.plot(iteration_times, elbo_trace, marker='o', linestyle='-', color='b', label="ELBO")
    #
    # # Ensure valid values before calling fill_between
    # if np.all(np.isfinite(elbo_trace)) and np.all(np.isfinite(elbo_sd_trace)):
    #     plt.fill_between(iteration_times, elbo_trace - elbo_sd_trace, elbo_trace + elbo_sd_trace, color='b', alpha=0.2,
    #                      label="ELBO ± 1 SD")
    # else:
    #     print("Warning: ELBO trace contains invalid values, skipping shaded area.")
    #
    # plt.xlabel("Iteration")
    # plt.ylabel("ELBO")
    # plt.title("PyVBMC ELBO Convergence Trace")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    if task.name == 'svar' and not noisy_likelihood:
        task.set_raw(False)


    total_simulations = vbmc.iteration_history["func_count"][-1] * simulations_per_eval if noisy_likelihood else None
    # if noisy_likelihood:
    #     print(f"noise: {vbmc.function_logger.S}")
    #     print(f"number of GP training points: {len(vbmc.gp.X)}")
    return torch.tensor(posterior_samples, dtype=torch.float32), total_simulations,vbmc
