import numpy as np
from scipy.stats import norm


def gaussian_copula_sample(
    soilph_min, soilph_max,
    duration_min, duration_max,
    temp_min, temp_max,
    water_min, water_max,
    humidity_min, humidity_max,
    n_min, n_max,
    p_min, p_max,
    k_min, k_max,
    n_samples,
    seed=42
):
    """
    Generate correlated synthetic samples for all numerical features
    using Gaussian Copula.
    """

    np.random.seed(seed)

    # Correlation matrix for 8 numerical features
    correlation = np.array([
        [1.00, 0.35, 0.40, 0.30, 0.30, 0.20, 0.20, 0.20],
        [0.35, 1.00, 0.45, 0.40, 0.35, 0.30, 0.25, 0.25],
        [0.40, 0.45, 1.00, 0.55, 0.45, 0.30, 0.25, 0.25],
        [0.30, 0.40, 0.55, 1.00, 0.50, 0.35, 0.30, 0.30],
        [0.30, 0.35, 0.45, 0.50, 1.00, 0.35, 0.30, 0.30],
        [0.20, 0.30, 0.30, 0.35, 0.35, 1.00, 0.60, 0.50],
        [0.20, 0.25, 0.25, 0.30, 0.30, 0.60, 1.00, 0.55],
        [0.20, 0.25, 0.25, 0.30, 0.30, 0.50, 0.55, 1.00]
    ])

    mean = np.zeros(8)

    # Generate correlated normal variables
    z = np.random.multivariate_normal(
        mean,
        correlation,
        size=n_samples
    )

    # Convert to Uniform(0,1)
    u = norm.cdf(z)

    # Scale each feature to its crop-specific range
    soil_ph = soilph_min + u[:, 0] * (soilph_max - soilph_min)

    duration = duration_min + u[:, 1] * (duration_max - duration_min)

    temperature = temp_min + u[:, 2] * (temp_max - temp_min)

    water = water_min + u[:, 3] * (water_max - water_min)

    humidity = humidity_min + u[:, 4] * (humidity_max - humidity_min)

    N = n_min + u[:, 5] * (n_max - n_min)

    P = p_min + u[:, 6] * (p_max - p_min)

    K = k_min + u[:, 7] * (k_max - k_min)

    return (
        soil_ph,
        duration,
        temperature,
        water,
        humidity,
        N,
        P,
        K
    )