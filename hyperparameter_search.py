"""Small dependency-free randomized search spaces for the notebook benchmark."""

from __future__ import annotations

import numpy as np


SEARCH_SPACE = {
    "TimeGAN": {
        "hidden_dim": [16, 24, 32], "num_layers": [2, 3], "gamma": [0.5, 1.0, 2.0],
        "epochs": [50, 100, 200], "batch_size": [32, 64], "lr": [3e-4, 1e-3],
    },
    "TimeVAE": {
        "latent_dim": [4, 8, 16], "reconstruction_weight": [1.0, 3.0, 5.0],
        "epochs": [50, 100, 200], "batch_size": [32, 64], "lr": [3e-4, 1e-3],
    },
    "DiffusionTS": {
        "d_model": [32, 64, 96], "n_heads": [2, 4], "n_layers": [2, 3],
        "timesteps": [100, 200, 300], "epochs": [50, 100, 200],
        "batch_size": [32, 64], "lr": [3e-5, 1e-4, 3e-4],
    },
    "StudentTARIMAGARCH": {"order": [(0, 0, 0), (1, 0, 0), (1, 0, 1)]},
    "StableGARCH": {"tail_fraction": [0.05, 0.1, 0.2]},
}


def sample_hyperparameters(model_name, rng=None):
    """Draw one reproducible candidate from the declared discrete search space."""
    rng = np.random.default_rng() if rng is None else rng
    if model_name not in SEARCH_SPACE:
        raise KeyError(f"No search space registered for {model_name!r}")
    return {name: values[int(rng.integers(len(values)))]
            for name, values in SEARCH_SPACE[model_name].items()}


def random_search(model_name, evaluate_candidate, n_trials=5, seed=0):
    """Run randomized search using a caller-supplied validation objective.

    The callback receives a candidate dict and returns a scalar loss (lower is
    better). Failed candidates are retained in the trial table with their error.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least one")
    rng = np.random.default_rng(seed)
    trials, seen = [], set()
    max_unique = int(np.prod([len(v) for v in SEARCH_SPACE[model_name].values()]))
    for trial_id in range(min(n_trials, max_unique)):
        candidate = sample_hyperparameters(model_name, rng)
        key = tuple((k, repr(v)) for k, v in sorted(candidate.items()))
        if key in seen:
            continue
        seen.add(key)
        try:
            loss = float(evaluate_candidate(candidate))
            if not np.isfinite(loss):
                raise ValueError("validation objective returned a non-finite score")
            trials.append({"trial": trial_id, "params": candidate, "loss": loss, "error": None})
        except Exception as exc:  # keep the search running if one setting is invalid
            trials.append({"trial": trial_id, "params": candidate, "loss": np.nan,
                           "error": f"{type(exc).__name__}: {exc}"})
    valid = [row for row in trials if np.isfinite(row["loss"])]
    if not valid:
        raise RuntimeError(f"All {len(trials)} {model_name} tuning trials failed")
    best = min(valid, key=lambda row: row["loss"])
    return best["params"], trials
