import logging
import numpy as np
import torch
from typing import Optional

from pyvbmc import VBMC
from sbibm.tasks.task import Task

def run(
    task: Task,
    num_samples: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
) -> (torch.Tensor, int, Optional[torch.Tensor]):
    """Runs PyVBMC to perform variational Bayesian inference.

    Args:
        task: Task instance
        num_samples: Number of samples to generate from posterior
        num_observation: Observation number to load, alternative to `observation`
        observation: Observation, alternative to `num_observation`

    Returns:
        Samples from the variational posterior, number of function evaluations, and ELBO.
    """
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    log.info("Running PyVBMC for approximate Bayesian inference.")

    if task.name == 'svar':
        task.set_raw(True)



    log_prob_fn = task._get_log_prob_fn(num_observation=num_observation, posterior=True)


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
        PLB = np.maximum(prior_mean - np.sqrt(np.diag(prior_cov)), 1e-6)  # 确保 PLB > 0
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
        return log_prob_fn(torch.tensor(theta, dtype=torch.float32)).numpy().flatten()

    # run PyVBMC
    vbmc = VBMC(log_prob_wrapper, x0, LB, UB, PLB, PUB)


    # vbmc = VBMC(log_prob_fn, x0, LB, UB, PLB, PUB)
    vp, results = vbmc.optimize()

    # sample posterior
    posterior_samples,_ = vp.sample(num_samples)

    # calculate ELBO
    elbo = results["elbo"]

    # todo: posterior for true params
    true_params = task.get_true_parameters(num_observation=num_observation).numpy().reshape(1, -1)

    log_prob_true_pyvbmc = vp.log_pdf(true_params)
    print(f"true params: {true_params}")
    print(f"pyvbmc posterior mean: {np.mean(posterior_samples,axis=0)}")
    print(f"prob true params pyvbmc: {np.exp(log_prob_true_pyvbmc)}")


    if task.name == 'svar':
        task.set_raw(False)


    return torch.tensor(posterior_samples, dtype=torch.float32), log_prob_true_pyvbmc, elbo
