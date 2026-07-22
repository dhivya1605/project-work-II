from scipy.stats import qmc
import numpy as np

def lhs_sample(min_value, max_value, n_samples, seed=42):
    """
    Generate samples using Latin Hypercube Sampling.
    """

    # Handle equal min and max values
    if min_value == max_value:
        return np.full(n_samples, min_value)

    sampler = qmc.LatinHypercube(d=1, seed=seed)

    sample = sampler.random(n=n_samples)

    scaled = qmc.scale(
        sample,
        [min_value],
        [max_value]
    )

    return scaled.flatten()