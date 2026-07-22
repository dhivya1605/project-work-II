import numpy as np
from scipy.stats import qmc

def sobol_sample(min_value, max_value, n_samples, seed=42):
    """
    Generate samples using Sobol Quasi-Random Sampling.
    """

    # Handle equal values
    if min_value == max_value:
        return np.full(n_samples, min_value)

    sampler = qmc.Sobol(
        d=1,
        scramble=True,
        seed=seed
    )

    # Sobol requires power of 2
    m = int(np.ceil(np.log2(n_samples)))

    sample = sampler.random_base2(m=m)

    sample = sample[:n_samples]

    scaled = qmc.scale(
        sample,
        [min_value],
        [max_value]
    )

    return scaled.flatten()