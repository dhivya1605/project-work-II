"""
dlo.py
------
Draco Lizard Optimizer (DLO) (Wang, X., "Draco lizard optimizer: a novel
metaheuristic algorithm for global optimization problems", Evolutionary
Intelligence, 2025), adapted here for continuous hyperparameter tuning of
TabKANet.

DLO models the Draco lizard's gliding and adaptive survival behavior as two
phases over a population of candidate solutions:
  1. Gliding (exploration) phase - each lizard "glides" from its current
     perch toward a randomly chosen better position in the population,
     covering large distances early in the search (long, shallow glides).
  2. Perching / adaptive refinement (exploitation) phase - lizards make
     small adjustments around the best-known perch (the global best),
     with the step size shrinking as iterations progress (steeper, shorter
     glides as the lizard approaches its target branch).

A gliding-ratio control parameter interpolates between the two phases across
iterations, mirroring how the published algorithm balances exploration and
exploitation.

Here each "position" is a real-valued vector encoding TabKANet's
hyperparameters (embedding dim, number of attention heads, number of
transformer layers, feed-forward dim, dropout, learning rate), each mapped
into a fixed [0, 1] range and decoded via `decode_hyperparameters`.
"""

import numpy as np


# Search space bounds for each hyperparameter (used for decoding [0,1] -> real value)
HYPERPARAM_SPACE = {
    "embed_dim":        {"type": "int",   "choices": [16, 24, 32, 48, 64]},
    "n_heads":          {"type": "int",   "choices": [2, 4, 8]},
    "n_transformer_layers": {"type": "int", "choices": [1, 2, 3, 4]},
    "ff_dim":           {"type": "int",   "choices": [32, 64, 96, 128]},
    "dropout":          {"type": "float", "low": 0.0, "high": 0.4},
    "learning_rate":    {"type": "float", "low": 1e-4, "high": 5e-3, "log": True},
}
PARAM_NAMES = list(HYPERPARAM_SPACE.keys())
DIM = len(PARAM_NAMES)


def decode_hyperparameters(position: np.ndarray) -> dict:
    """Map a position vector in [0,1]^DIM to a concrete hyperparameter dict.
    NOTE: embed_dim must stay divisible by n_heads for the Transformer, so we
    snap n_heads down to the largest valid divisor after decoding.
    """
    params = {}
    for i, name in enumerate(PARAM_NAMES):
        spec = HYPERPARAM_SPACE[name]
        val = np.clip(position[i], 0.0, 1.0)

        if spec["type"] == "int":
            choices = spec["choices"]
            idx = min(int(val * len(choices)), len(choices) - 1)
            params[name] = choices[idx]
        else:  # float
            low, high = spec["low"], spec["high"]
            if spec.get("log"):
                log_low, log_high = np.log10(low), np.log10(high)
                params[name] = float(10 ** (log_low + val * (log_high - log_low)))
            else:
                params[name] = float(low + val * (high - low))

    # Ensure embed_dim % n_heads == 0
    valid_heads = [h for h in HYPERPARAM_SPACE["n_heads"]["choices"] if params["embed_dim"] % h == 0]
    if params["n_heads"] not in valid_heads:
        params["n_heads"] = max(valid_heads) if valid_heads else 1

    return params


class DLOHyperparameterOptimizer:
    def __init__(
        self,
        population_size: int = 12,
        max_iterations: int = 15,
        random_state: int = 42,
    ):
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.rng = np.random.default_rng(random_state)

    def optimize(self, fitness_fn, verbose: bool = True):
        """
        fitness_fn: callable(hyperparams_dict) -> float (LOWER is better,
        e.g. validation loss or 1 - val_accuracy). This should train
        TabKANet briefly (few epochs) and return a validation metric.
        """
        population = self.rng.random((self.population_size, DIM))
        fitness = np.array([
            fitness_fn(decode_hyperparameters(population[i]))
            for i in range(self.population_size)
        ])

        best_idx = np.argmin(fitness)
        best_pos = population[best_idx].copy()
        best_fit = fitness[best_idx]
        history = [best_fit]

        for it in range(self.max_iterations):
            # Gliding ratio: starts near 1 (mostly exploration/gliding),
            # decays toward 0 (mostly perching/exploitation) over iterations
            glide_ratio = 1.0 - (it / max(1, self.max_iterations - 1))

            for i in range(self.population_size):
                r = self.rng.random(DIM)

                if self.rng.random() < glide_ratio:
                    # ---- Gliding (exploration) phase ----
                    partner_idx = self.rng.integers(0, self.population_size)
                    partner = population[partner_idx]
                    glide_distance = 2.0 * r - 1.0  # in [-1, 1]
                    new_pos = population[i] + glide_distance * glide_ratio * (partner - population[i])
                else:
                    # ---- Perching / adaptive refinement (exploitation) phase ----
                    step = (1 - glide_ratio) * (2 * r - 1) * 0.3
                    new_pos = best_pos + step

                new_pos = np.clip(new_pos, 0.0, 1.0)
                new_fit = fitness_fn(decode_hyperparameters(new_pos))

                if new_fit < fitness[i]:
                    population[i] = new_pos
                    fitness[i] = new_fit

            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fit:
                best_fit = fitness[gen_best_idx]
                best_pos = population[gen_best_idx].copy()

            history.append(best_fit)
            if verbose:
                print(f"[DLO] iter {it + 1:02d}/{self.max_iterations} "
                      f"best_fitness={best_fit:.4f} best_params={decode_hyperparameters(best_pos)}")

        self.best_position_ = best_pos
        self.best_hyperparameters_ = decode_hyperparameters(best_pos)
        self.best_fitness_ = best_fit
        self.history_ = history
        return self.best_hyperparameters_
