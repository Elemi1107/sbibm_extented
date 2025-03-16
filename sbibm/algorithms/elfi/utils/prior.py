import logging

import elfi
import numpy as np
import scipy.stats
from scipy.stats import uniform, rv_continuous

from sbibm.tasks.task import Task
from torch.distributions import Uniform, TransformedDistribution
from torch.distributions.transforms import SigmoidTransform,TanhTransform



def build_prior(task: Task, model: elfi.ElfiModel):
    log = logging.getLogger(__name__)
    log.warn("Will discard any correlations in prior")

    bounds = {}

    prior_cls = str(task.prior_dist)
    if prior_cls == "Independent()":
        prior_cls = str(task.prior_dist.base_dist)

    prior_params = {}
    if "MultivariateNormal" in prior_cls:
        prior_params["m"] = task.prior_params["loc"].numpy()
        if "precision_matrix" in prior_cls:
            prior_params["C"] = np.linalg.inv(
                task.prior_params["precision_matrix"].numpy()
            )
        if "covariance_matrix" in prior_cls:
            prior_params["C"] = task.prior_params["covariance_matrix"].numpy()

        for dim in range(task.dim_parameters):
            loc = prior_params["m"][dim]
            scale = np.sqrt(prior_params["C"][dim, dim])

            elfi.Prior(
                "norm",
                loc,
                scale,
                model=model,
                name=f"parameter_{dim}",
            )

            bounds[f"parameter_{dim}"] = (
                prior_params["m"][dim] - 3.0 * np.sqrt(prior_params["C"][dim, dim]),
                prior_params["m"][dim] + 3.0 * np.sqrt(prior_params["C"][dim, dim]),
            )

    elif "Uniform" in prior_cls:
        prior_params["low"] = task.prior_params["low"].numpy()
        prior_params["high"] = task.prior_params["high"].numpy()

        for dim in range(task.dim_parameters):
            loc = prior_params["low"][dim]
            scale = prior_params["high"][dim] - loc

            elfi.Prior(
                "uniform",
                loc,
                scale,
                model=model,
                name=f"parameter_{dim}",
            )

            bounds[f"parameter_{dim}"] = (
                prior_params["low"][dim],
                prior_params["high"][dim],
            )

        if task.name == "svar":
            bounds[f"parameter_{task.dim_parameters-1}"] = (1e-6, 1) # avoid 0 variance

    elif "LogNormal" in prior_cls:
        prior_params["loc"] = task.prior_params["loc"].numpy()
        prior_params["scale"] = task.prior_params["scale"].numpy()

        for dim in range(task.dim_parameters):
            # loc = prior_params["loc"][dim]
            # scale = prior_params["scale"][dim]
            mu = prior_params["loc"][dim]
            sigma = max(prior_params["scale"][dim], 1e-6)

            elfi.Prior(
                "lognorm",
                sigma, # s = σ
                0, # loc = 0
                np.exp(mu),  # scale = exp(μ)
                model=model,
                name=f"parameter_{dim}",
            )


            bounds[f"parameter_{dim}"] = (
                np.exp(mu - 3.0 * sigma),
                np.exp(mu + 3.0 * sigma)
            )

        print(bounds)

    else:
        log.info("No support for prior yet")
        raise NotImplementedError

    return bounds
