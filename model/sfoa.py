"""
sfoa.py
-------
Superb Fairy-wren Optimization Algorithm (SFOA), adapted as a binary wrapper
feature-selection method (Jia et al., "Superb Fairy-wren Optimization
Algorithm: a novel metaheuristic algorithm for solving feature selection
problems", Cluster Computing, 2025).

SFOA models three behavioral phases of the superb fairy-wren:
  1. Growth phase      - juveniles (candidate solutions) grow toward the
                          best-known nest (global best) with an exploratory
                          random component.
  2. Breeding phase     - solutions recombine with other good solutions in
                          the population, mixing traits (exploitation).
  3. Predator-avoidance - a fraction of the population makes a large,
                          randomized escape jump to preserve diversity and
                          avoid getting stuck in local optima.

This implementation is a faithful *behavioral* reproduction of that
three-phase structure using a continuous position vector in [0, 1]^D that is
thresholded at 0.5 into a binary feature mask (standard approach for turning
any continuous metaheuristic into a wrapper feature selector).

Fitness function (minimized): a weighted combination of classification error
and the fraction of features used, so SFOA is pushed toward small, accurate
feature subsets:

    fitness = alpha * error_rate + (1 - alpha) * (num_selected / num_total)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class SFOAFeatureSelector:
    def __init__(
        self,
        population_size: int = 20,
        max_iterations: int = 30,
        alpha: float = 0.9,
        predator_fraction: float = 0.2,
        random_state: int = 42,
    ):
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.predator_fraction = predator_fraction
        self.rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------ #
    def _fitness(self, mask: np.ndarray, X_train, y_train, X_val, y_val) -> float:
        selected = np.where(mask > 0.5)[0]
        if len(selected) == 0:
            return 1.0  # worst possible fitness, no features selected

        clf = RandomForestClassifier(n_estimators=60, random_state=0, n_jobs=-1)
        clf.fit(X_train[:, selected], y_train)
        preds = clf.predict(X_val[:, selected])
        error_rate = 1.0 - (preds == y_val).mean()

        feature_ratio = len(selected) / mask.shape[0]
        return self.alpha * error_rate + (1 - self.alpha) * feature_ratio

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        n_samples, n_features = X.shape
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=0, stratify=y
        )

        # Initialize population: continuous positions in [0, 1]
        population = self.rng.random((self.population_size, n_features))

        fitness = np.array([
            self._fitness(population[i], X_train, y_train, X_val, y_val)
            for i in range(self.population_size)
        ])

        best_idx = np.argmin(fitness)
        best_pos = population[best_idx].copy()
        best_fit = fitness[best_idx]
        history = [best_fit]

        n_predators = max(1, int(self.predator_fraction * self.population_size))

        for it in range(self.max_iterations):
            t_ratio = it / max(1, self.max_iterations - 1)  # 0 -> 1 over the run

            for i in range(self.population_size):
                r1, r2, r3 = self.rng.random(3)

                if i % 3 == 0:
                    # ---- Growth phase: move toward global best ----
                    growth_rate = 1.0 - t_ratio  # juveniles grow fast early, settle later
                    new_pos = population[i] + growth_rate * r1 * (best_pos - population[i])

                elif i % 3 == 1:
                    # ---- Breeding phase: recombine with a random better peer ----
                    partner_idx = self.rng.integers(0, self.population_size)
                    partner = population[partner_idx]
                    new_pos = r2 * population[i] + (1 - r2) * partner

                else:
                    # ---- Predator-avoidance phase: escape jump ----
                    new_pos = population[i] + (2 * r3 - 1) * (1 - t_ratio) * self.rng.random(n_features)

                new_pos = np.clip(new_pos, 0.0, 1.0)
                new_fit = self._fitness(new_pos, X_train, y_train, X_val, y_val)

                if new_fit < fitness[i]:
                    population[i] = new_pos
                    fitness[i] = new_fit

            # Random escape jumps for a subset of the worst solutions
            # (predator-avoidance diversity injection, prevents premature convergence)
            worst_indices = np.argsort(fitness)[-n_predators:]
            for idx in worst_indices:
                population[idx] = self.rng.random(n_features)
                fitness[idx] = self._fitness(population[idx], X_train, y_train, X_val, y_val)

            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fit:
                best_fit = fitness[gen_best_idx]
                best_pos = population[gen_best_idx].copy()

            history.append(best_fit)
            if verbose:
                n_sel = int((best_pos > 0.5).sum())
                print(f"[SFOA] iter {it + 1:03d}/{self.max_iterations} "
                      f"best_fitness={best_fit:.4f} selected_features={n_sel}/{n_features}")

        self.best_mask_ = best_pos > 0.5
        self.best_position_ = best_pos
        self.best_fitness_ = best_fit
        self.history_ = history
        return self

    def get_selected_indices(self) -> np.ndarray:
        return np.where(self.best_mask_)[0]
