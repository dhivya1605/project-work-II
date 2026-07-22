import numpy as np
from scipy.stats import truncnorm

def truncated_normal_sample(min_value, max_value, n_samples, seed=42):
    """
    Generate samples using Truncated Normal Distribution.
    """

    # Handle equal values
    if min_value == max_value:
        return np.full(n_samples, min_value)

    mean = (min_value + max_value) / 2

    std = (max_value - min_value) / 6

    a = (min_value - mean) / std
    b = (max_value - mean) / std

    np.random.seed(seed)

    sample = truncnorm.rvs(
        a,
        b,
        loc=mean,
        scale=std,
        size=n_samples
    )

    return sample