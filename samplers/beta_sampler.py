import numpy as np
from scipy.stats import beta

def beta_sample(min_value, max_value, n_samples, seed=42):
    """
    Generate samples using Beta Distribution.
    """

    # Handle equal values
    if min_value == max_value:
        return np.full(n_samples, min_value)

    np.random.seed(seed)

    # Shape parameters
    alpha = 2
    beta_param = 2

    samples = beta.rvs(
        alpha,
        beta_param,
        size=n_samples,
        random_state=seed
    )

    # Scale to the required range
    samples = min_value + samples * (max_value - min_value)

    return samples