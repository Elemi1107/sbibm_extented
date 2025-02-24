import os

import sbibm
import numpy as np
import torch
import time
import pandas as pd
from joblib import Parallel, delayed
from sbibm.algorithms import bolfi, snpe, pyvbmc


RESULTS_DIR = "./experiment_results"
os.makedirs(RESULTS_DIR, exist_ok=True)



def get_posterior_samples_filename(task_name, algorithm, num_simulations):
    return os.path.join(RESULTS_DIR, f"{task_name}_{algorithm}_{num_simulations}.npz")

def load_posterior_samples(task_name, algorithm, num_simulations):
    filename = get_posterior_samples_filename(task_name, algorithm, num_simulations)
    if os.path.exists(filename):
        return torch.tensor(np.load(filename)["samples"])
    return None

def save_posterior_samples(task_name, algorithm, num_simulations, samples):
    filename = get_posterior_samples_filename(task_name, algorithm, num_simulations)
    np.savez(filename, samples=samples.numpy())

def run_experiment(task_name, algo_name, num_samples=10000, num_simulations=2000, num_observation=1):
    """ run experiments and  store posterior samples"""
    print(f"Running {algo_name} on {task_name} with {num_simulations} simulations...")

    # check if samples already exist
    posterior_samples = load_posterior_samples(task_name, algo_name, num_simulations)
    if posterior_samples is not None:
        print(f"Loaded cached results for {task_name} - {algo_name} - {num_simulations}")
        return posterior_samples

    task = sbibm.get_task(task_name)

    algorithm = algorithms[algo_name]
    # todo: record time
    # todo: adjust/get num_evaluations for pyvbmc
    # todo: return other features and store , especially convergence with time (fit time, sampling not included)
    posterior_samples, _, _ = algorithm(task=task, num_samples=1000, num_observation=num_observation,
                                        num_simulations=num_simulations)

    save_posterior_samples(task_name, algo_name, num_simulations, posterior_samples)

    return posterior_samples


# tasks = ["bernoulli_glm", "sir"]
# algorithms = {"pyvbmc": pyvbmc, "snpe": snpe, "bolfi": bolfi}
tasks = ["svar"]
algorithms = {"pyvbmc": pyvbmc}
num_simulations_list = [100, 1000, 2000]
num_observation = 1

for algorithm in algorithms.keys():
    for task_name in tasks:
        for num_simulations in num_simulations_list:
            run_experiment(task_name, algorithm, num_simulations=num_simulations)

# run parallel
# results = Parallel(n_jobs=-1)(
#     delayed(run_experiment)(task, algo, sim) for task in tasks for algo in algorithms for sim in num_simulations_list
# )

