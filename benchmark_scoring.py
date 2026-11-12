"""Benchmark-normalized composite and multi-objective ranking utilities.

Scores are computed within one experiment against an iid Gaussian reference
estimated from training data. Lower scores are better. The Gaussian reference
is a normalization baseline, not a guardrail or a claim of financial realism.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


SCORING_VERSION = "benchmark-normalized-worst-cluster-v1"
EPSILON = 1e-10
WORST_CLUSTER_PENALTY = 0.15

CLUSTER_WEIGHTS = {
    "distribution_tails": 0.35,
    "temporal_structure": 0.35,
    "multivariate_dependence": 0.20,
    "fidelity_and_utility": 0.10,
}

CLUSTER_METRICS = {
    "distribution_tails": [
        "hill_error", "skewness_error", "kurtosis_error",
        "var_error_asset", "var_error_portfolio",
    ],
    "temporal_structure": [
        "raw_return_acf_error", "volatility_clustering_error",
        "spectral_error", "leverage_effect_error",
    ],
    "multivariate_dependence": ["correlation_error"],
    "fidelity_and_utility": ["discriminative_score", "predictive_score"],
}

# User-selected lexicographic priorities: distribution, downstream usefulness,
# then remaining criteria in this stated order. Values are baseline-normalized.
LEXICOGRAPHIC_ORDER = [
    "distribution_tails",
    "downstream_usefulness",
    "temporal_structure",
    "multivariate_dependence",
    "discriminative_score",
]


def sample_gaussian_reference(
    training_observations: np.ndarray,
    n_windows: int,
    window_size: int,
    seed: int,
) -> np.ndarray:
    """Sample iid multivariate-normal windows using training-only moments."""
    values = np.asarray(training_observations, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("training_observations must have shape (n_observations, n_features)")
    if n_windows < 1 or window_size < 2:
        raise ValueError("n_windows must be positive and window_size must be >= 2")
    if not np.isfinite(values).all():
        raise ValueError("training_observations contains non-finite values")

    mean = values.mean(axis=0)
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    # Small ridge makes the covariance usable for constant/collinear features.
    scale = max(float(np.trace(covariance)) / covariance.shape[0], 1e-12)
    covariance = covariance + np.eye(covariance.shape[0]) * scale * 1e-10
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mean, covariance, size=n_windows * window_size)
    return draws.reshape(n_windows, window_size, values.shape[1])


def metric_error(metric_name: str, value: float) -> float:
    """Map metric values to non-negative errors, including predictive ratio."""
    value = float(value)
    if not np.isfinite(value):
        return np.nan
    if metric_name == "predictive_score":
        # Ratio is ideal at one; symmetric multiplicative deviations are equal.
        return abs(float(np.log(max(value, EPSILON))))
    return max(value, 0.0)


def normalized_metric_error(metric_name: str, value: float, baseline: float) -> float:
    model_error = metric_error(metric_name, value)
    baseline_error = metric_error(metric_name, baseline)
    if not np.isfinite(model_error) or not np.isfinite(baseline_error):
        return np.nan
    return float(model_error / (model_error + baseline_error + EPSILON))


def _cluster_score(
    metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
    names: Sequence[str],
) -> float:
    values = [normalized_metric_error(name, metrics.get(name, np.nan),
                                      baseline_metrics.get(name, np.nan))
              for name in names]
    if not values or not np.isfinite(values).all():
        return np.nan
    return float(np.mean(values))


@dataclass(frozen=True)
class ScoreResult:
    composite_score: float
    average_score: float
    worst_cluster_score: float
    cluster_scores: Mapping[str, float]
    normalized_metric_scores: Mapping[str, float]
    lexicographic_scores: Mapping[str, float]


def score_model(
    metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
    multivariate: bool,
) -> ScoreResult:
    """Compute the friend's composite and lexicographic objective components.

    The univariate case omits correlation and portfolio VaR (the latter would
    duplicate asset VaR) and renormalizes the active cluster weights.
    """
    clusters = {name: list(names) for name, names in CLUSTER_METRICS.items()}
    if not multivariate:
        clusters["distribution_tails"].remove("var_error_portfolio")
        clusters.pop("multivariate_dependence")

    normalized = {
        metric: normalized_metric_error(metric, metrics.get(metric, np.nan),
                                        baseline_metrics.get(metric, np.nan))
        for names in clusters.values() for metric in names
    }
    scores = {name: _cluster_score(metrics, baseline_metrics, names)
              for name, names in clusters.items()}

    active_weight = sum(CLUSTER_WEIGHTS[name] for name in scores)
    weights = {name: CLUSTER_WEIGHTS[name] / active_weight for name in scores}
    if all(np.isfinite(v) for v in scores.values()):
        average = float(sum(weights[name] * scores[name] for name in scores))
        worst = float(max(scores.values()))
        composite = ((1 - WORST_CLUSTER_PENALTY) * average
                     + WORST_CLUSTER_PENALTY * worst)
    else:
        average = worst = composite = np.nan

    lex_scores = {
        "distribution_tails": scores["distribution_tails"],
        "downstream_usefulness": normalized.get("predictive_score", np.nan),
        "temporal_structure": scores["temporal_structure"],
        "multivariate_dependence": scores.get("multivariate_dependence", np.nan),
        "discriminative_score": normalized.get("discriminative_score", np.nan),
    }
    return ScoreResult(composite, average, worst, scores, normalized, lex_scores)
