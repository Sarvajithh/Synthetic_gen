# Benchmark ranking protocol

## Project scores

`benchmark_scoring.py` implements the supplied benchmark-normalized composite. For every experiment, an iid multivariate-Gaussian reference is sampled from training-only moments and evaluated against the same validation or test windows as each candidate. Each metric error is normalized as `e / (e + b + epsilon)`, where `e` is the candidate error and `b` is the Gaussian-reference error. Predictive ratios are first transformed to `abs(log(ratio))`, making one the ideal ratio.

The four cluster weights are distribution/tails 0.35, temporal structure 0.35, multivariate dependence 0.20, and fidelity/utility 0.10. The final composite is `0.85 * weighted_mean + 0.15 * worst_cluster`. For univariate SPY, correlation and portfolio VaR are omitted because correlation is undefined/use-less with one channel and portfolio VaR duplicates the asset VaR; active weights are renormalized. This is a ranking heuristic, not an acceptance threshold.

The notebook also reports a Pareto front over the available cluster scores and lexicographic order: distribution/tails, downstream predictive usefulness, temporal structure, multivariate dependence (when applicable), then discriminative score. No guardrails are used. A model's individual metric values remain in the raw per-experiment table.

## Literature comparison

`rank_results.py` writes a literature source table, source-level ranks, a normalized cross-paper rank summary, and a join to project model results. Literature aggregate ranks exclude StableGARCH because no source in the selected comparable set provides a StableGARCH result. TimeGAN and TimeVAE are the only model-name overlaps used in the exploratory aggregate. Model-name overlap does not guarantee identical implementations.

- TSGBench reports average ranks of 3.83 for TimeVAE and 6.31 for TimeGAN across ten models, datasets, and measures.
- *Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting* (NeurIPS 2023) compares TimeVAE and TimeGAN on downstream CRPS over eight datasets and three evaluator models. The paired average ranks in this project are derived from 16/24 lower-CRPS comparisons for TimeVAE and 8/24 for TimeGAN. TSDiff itself is not equated with this project's DiffusionTS.
- *Generative Learning for Financial Time Series with Irregular and Scale-Invariant Patterns* (ICLR 2024) and the 2025 financial diffusion comparison have only one direct project model-name overlap (TimeGAN), so they are documented but do not contribute a rank to the cross-paper aggregate.

The cross-paper aggregate averages each contributing paper's normalized rank percentile with equal paper weight. It is descriptive only: the papers have different domains, metrics, dataset candidate sets, and task definitions. It must not be interpreted as an industry leaderboard or a controlled head-to-head result.

## Outputs

Running the notebook writes these files under `results_v2/`:

- `rankings_per_experiment.csv`: every generator's scores and three rankings for each data configuration, frequency, and training duration.
- `rankings_by_segment.csv`: mean ranks across training durations, separated by data configuration and frequency.
- `literature_sources.csv`, `literature_reference_ranks.csv`, and `literature_aggregate_ranks.csv`.
- `project_vs_literature.csv`: project/literature rows for overlapping model names only.

A project-versus-literature comparison is contextual. Directly comparable claims require the same data, model implementation, split, and metric definition.
