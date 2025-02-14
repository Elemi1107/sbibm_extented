import random

import pandas as pd
import torch
import numpy as np
import pymc as pm
import arviz as az
from pathlib import Path
from typing import Callable, List, Optional, Any
import pyro
from pyro import distributions as pdist

from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task


class SVAR(Task):
    def __init__(self, num_observations: int = 1, k: int = 6, T: int = 1000):
        """Sparse Vector Autoregression (SVAR) Task

        Args:
            k: Dimension of the time series (default: 6)
            T: Number of observations (default: 1000)
        """
        self.k_list = list(range(6,25,2))
        self.T = T
        self.k = self.k_list[num_observations - 1] # different k per observation
        dim_parameters = self.k + 1  # k off-diagonal elements + variance parameter
        dim_data = self.k + 1  # Summary statistics, each param one stats
        # self.pairs = self._load_pairs(num_observations)
        self.pairs = None # load dynamically

        observation_seeds = [1000000 + i for i in range(10)]  # Fixed observation seeds

        super().__init__(
            dim_data=dim_data,
            dim_parameters=dim_parameters,
            name="svar",
            num_observations=num_observations,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
            observation_seeds=observation_seeds,
        )



        # Prior: Uniform distribution for all parameters
        self.prior_params = {
            "low": torch.cat([-torch.ones(self.k), torch.tensor([0.01])]),  # Last param is variance
            "high": torch.ones(self.k + 1)
        }
        self.prior_dist = pdist.Independent(
            pdist.Uniform(self.prior_params["low"], self.prior_params["high"]),
            reinterpreted_batch_ndims=1
        )
        self.prior_dist.set_default_validate_args(False)

    def _load_pairs(self, num_observation: int):
        """
        Load the parameter locations (pairs) from the true parameters file.
        Each (i, j) appears twice with separate values.
        """
        path = self.path / f"files/num_observation_{num_observation}/true_parameters.csv"
        df = pd.read_csv(path)
        df_X = df[df["i"] != -1]
        pairs = [(int(row["i"]), int(row["j"])) for _, row in df_X.iterrows()]
        # values = torch.tensor(df_X["parameter"].values, dtype=torch.float32)

        return pairs  # return pairs tuple list + params values

    def set_k(self, num_observation: int):
        """Set `k` dynamically based on `num_observation`."""
        self.k = self.k_list[num_observation - 1]
        self.dim_data = self.k + 1
        self.dim_parameters = self.k + 1

    def get_prior(self) -> Callable:
        """Returns a function sampling from the prior."""
        self.set_k(self.num_observations)

        self.prior_params = {
            "low": torch.cat([-torch.ones(self.k), torch.tensor([0.01])]),
            "high": torch.ones(self.k + 1),
        }
        self.prior_dist = pdist.Independent(
            pdist.Uniform(self.prior_params["low"], self.prior_params["high"]),
            reinterpreted_batch_ndims=1
        )

        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def get_simulator(self, max_calls: Optional[int] = None) -> Simulator:
        """Returns a function that simulates SVAR observations given parameters."""

        def simulator(parameters):
            num_samples = parameters.shape[0]
            summaries = torch.zeros((num_samples, self.dim_parameters), dtype=torch.float32)

            for i in range(num_samples):
                summaries[i] = self.simulate_SVAR(parameters[i])

            return summaries

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)



    def simulate_SVAR(self, theta):
        """Simulates SVAR data and computes summary statistics."""
        if self.pairs is None:
            self.pairs = self._load_pairs(self.num_observations)

        X = -0.1 * torch.eye(self.k)

        matrix_params = theta[:-1]
        # Set off-diagonal elements based on parameter locations
        for (i,j), value in zip(self.pairs,matrix_params):
            X[i,j] = value

        # todo: sigma的logit转换？
        sigma = theta[-1]
        Y = torch.zeros(self.k, self.T)
        Y[:, 0] = torch.normal(0.0, sigma, size=(self.k,))

        for t in range(1, self.T):
            Y[:, t] = torch.matmul(X, Y[:, t - 1]) + torch.normal(0.0, sigma, size=(self.k,))

        return self.compute_summary(Y)

    def compute_summary(self, Y):
        """Computes summary statistics (autocovariance and standard deviation)."""
        S = torch.zeros(self.dim_parameters, dtype=torch.float32)

        for ii, (i,j) in enumerate(self.pairs):
            S[ii] = torch.mean(Y[i, 1:] * Y[j, :-1])

        S[-1] = torch.std(Y)
        return S

    def _sample_reference_posterior(
            self, num_samples: int, num_observation: Optional[int] = None, observation: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Uses PyMC to sample from the reference posterior using summary statistics."""
        if num_observation is not None:
            observation = self.get_observation(num_observation)

        k = self.k
        pairs = self.pairs if self.pairs is not None else self._load_pairs(num_observation)  # 获取当前 observation 的非零参数位置
        Y = observation.numpy()
        k, T = Y.shape

        with pm.Model() as model:
            theta = pm.Uniform("theta",
                               lower=self.prior_params["low"].numpy(),
                               upper=self.prior_params["high"].numpy(),
                               shape=(self.dim_parameters,))
            matrix_params = theta[:-1]
            sigma = theta[-1]

            # 在 PyMC 里构造 X
            X = -0.1 * np.eye(k)
            for (i, j), value in zip(pairs, matrix_params):
                X[i, j] = value # todo: 怎么构造X

            Y_pred = pm.math.dot(Y[:, :-1].T, X.T) # X*y_{t-1}

            # Likelihood?
            pm.Normal("y_obs", mu=Y_pred, sigma=sigma, observed=Y[:,1:].T)
            # todo: 从true params开始采样
            trace = pm.sample(draws=num_samples, tune=1000, chains=2, return_inferencedata=True)


        posterior_samples = az.extract(trace, var_names=["theta"]).values
        return torch.tensor(posterior_samples, dtype=torch.float32)


    def _generate_sparse_pairs(self, k):
        """
        Generate `k` locations of non-zero elements in X, ensuring (i,j) format.
        Only used when generating observation-true param pairs.
        """
        upper_pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        random.shuffle(upper_pairs)
        print(k)
        selected_pairs = upper_pairs[:k // 2]

        selected_pairs = selected_pairs + [(j,i) for (i,j) in selected_pairs]
        return selected_pairs

    def _save_true_parameters(self, num_observation: int, true_parameters: torch.Tensor, pairs):
        """Save true parameters, ensuring each (i,j) pair gets two separate entries."""
        path = (
                self.path / "files" / f"num_observation_{num_observation}" / "true_parameters.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        print(pairs)
        print(true_parameters.shape)
        for ii, (i, j) in enumerate(pairs):
            data.append([i, j, true_parameters[ii].item()])

        data.append([-1, -1, true_parameters[-1]])  # (-1, -1) 表示 `sigma`
        df = pd.DataFrame(data, columns=["i", "j", "parameter"])
        df.to_csv(path, index=False)

    def _setup(self,  create_reference: bool = True, **kwargs: Any):
        """Setup the task: generate observations and reference posterior samples"""

        for num_observation, observation_seed in enumerate(self.observation_seeds, start=1):
            np.random.seed(observation_seed)
            torch.manual_seed(observation_seed)

            print(f"Running setup for observation {num_observation} (seed={observation_seed})")

            self._save_observation_seed(num_observation, observation_seed)
            self.k = self.k_list[num_observation - 1]
            self.pairs = self._generate_sparse_pairs(self.k)  # FIXED pairs per observation
            self.num_observations = num_observation

            prior = self.get_prior()
            true_parameters = prior(num_samples=1)
            print("true params: ")
            print(true_parameters)
            self._save_true_parameters(num_observation, true_parameters.flatten(), self.pairs)

            simulator = self.get_simulator()
            observation = simulator(true_parameters)
            print(observation)
            self._save_observation(num_observation, observation)

            if create_reference:
                reference_posterior_samples = self._sample_reference_posterior(
                    num_observation=num_observation,
                    num_samples=self.num_reference_posterior_samples,
                    **kwargs,
                )
                self._save_reference_posterior_samples(num_observation, reference_posterior_samples)


if __name__ == "__main__":
    task = SVAR()
    task._sample_reference_posterior(num_samples=task.num_posterior_samples,num_observation=1)