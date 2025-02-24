import sbibm

task = sbibm.get_task("svar",num_observations=1)
# task = sbibm.get_task("bernoulli_glm")  # See sbibm.get_available_tasks() for all tasks
# task = sbibm.get_task("sir",summary="subsample")
# prior = task.get_prior()
# simulator = task.get_simulator()
# observation = task.get_observation(num_observation=1)  # 10 per task

# These objects can then be used for custom inference algorithms, e.g.
# we might want to generate simulations by sampling from prior:
# thetas = prior(num_samples=10_000)
# xs = simulator(thetas)

# Alternatively, we can import existing algorithms, e.g:
from sbibm.algorithms import bolfi # See help(rej_abc) for keywords
# posterior_samples, _, _ = bolfi(task=task, num_samples=1000, num_observation=1, num_simulations=100)
from sbibm.algorithms import pyvbmc
# posterior_samples, _, _ = pyvbmc(task=task, num_samples=100, num_observation=1)

from sbibm.algorithms import snpe
posterior_samples, _, _ = snpe(task=task, num_samples=1000, num_simulations=2000, num_observation=1)

# Once we got samples from an approximate posterior, compare them to the reference:
from sbibm.metrics import c2st
reference_samples = task.get_reference_posterior_samples(num_observation=1)
c2st_accuracy = c2st(reference_samples, posterior_samples)

# Visualise both posteriors:
from sbibm.visualisation import fig_posterior
fig = fig_posterior(task_name="sir", observation=1, samples=[posterior_samples])
# Note: Use fig.show() or fig.save() to show or save the figure
fig.show()
# # Get results from other algorithms for comparison:
# from sbibm.visualisation import fig_metric
# results_df = sbibm.get_results(dataset="main_paper.csv")
# fig = fig_metric(results_df.query("task == 'sir'"), metric="C2ST")