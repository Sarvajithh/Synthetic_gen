# Benchmark ranking protocol

## Project scoring methods

`main_v2.ipynb` reports three complementary project rankings for every experiment:

1. **Benchmark-Normalized Composite Score** — the proposed weighted score, renamed from the informal friend's-score label. Each metric error is normalized against an iid Gaussian reference estimated from the training data; predictive ratios are transformed with `abs(log(ratio))`. The clusters are distribution/tails (35%), temporal structure (35%), dependence (20%), and fidelity/utility (10%), with a 15% worst-cluster penalty. Lower is better. In univariate runs, correlation and portfolio VaR are omitted and remaining weights renormalized.
2. **Pareto Front** — nondominated sorting over the available cluster scores, all minimized. Front 1 contains the models for which no other model is at least as good in every included objective and better in one. Higher-numbered fronts are dominated by a preceding front. Pareto fronts preserve trade-offs rather than collapsing them into one scalar.
3. **Lexicographic Rank** — metrics are sorted by the declared priority: distribution/tails, downstream predictive usefulness, temporal structure, multivariate dependence when applicable, then discriminative score. An earlier component takes priority over every later component.

These methods are complementary views, not acceptance criteria. The Gaussian reference is a normalization baseline, not a realism target or pass/fail threshold. All raw metrics remain in the result tables.

## Literature comparison

`rank_results.py` keeps literature ranks **source-specific**. It does not average or otherwise pool them into a universal literature leaderboard: datasets, measures, candidate generators, and rank definitions differ across papers. Reported values are labeled as reported; values calculated from paper tables are labeled derived and include their basis and candidate-set size. Model family/architecture matches are explicitly marked as analogues.

The comparison covers all five project generators: TimeGAN, TimeVAE, DiffusionTS, StudentTARIMAGARCH, and StableGARCH. A blank source rank means no rank for that project model was established in that source. In particular, StableGARCH has no comparative generator rank in the selected sources. This is preferable to inventing a rank from a methods paper.

The sourced reference set includes TSGBench (reported TimeGAN and TimeVAE average ranks), the NeurIPS TSDiff paper (derived paired TimeGAN/TimeVAE ranks on downstream CRPS), the AAAI PaD-TS paper (derived values for Diffusion-TS, TimeGAN, and TimeVAE on Google Stocks), and a financial-data preprint (TimeGAN with generic-VAE and ARIMA-GARCH analogues). TSDiff is not treated as this project's DiffusionTS. The preprint and analogue ranks are contextual, not head-to-head claims about the project's implementations.

## Outputs

Running the notebook writes the following under `results_v2/`:

- `rankings_per_experiment.csv`: per-model metrics, composite score and rank, Pareto front, lexicographic rank for each experiment.
- `rankings_by_segment.csv`: means/ranks across training durations, separated by data configuration and frequency.
- `scores_and_rankings.csv` and `project_vs_literature.csv`: every project model row with all three project ranking methods, source-specific literature-rank columns, and coverage caveats. No model is dropped merely because literature did not rank it.
- `literature_sources.csv`, `literature_reference_ranks.csv`, and `literature_model_coverage.csv`: sources, source-level rank evidence, and explicit coverage for all five models.
- `literature_aggregate_ranks.csv`: a clear note that cross-paper pooled ranks are not computed because the studies are not comparable enough for a universal aggregate.

The distribution plot uses common histogram bins over the pooled central 0.5th–99.5th percentile range, plus a separate full-range absolute-return tail-survival panel. This keeps the center readable without hiding extreme values.

Direct literature-versus-project claims require matching data, implementation, split, and metric definitions; these tables provide context, not an industry leaderboard.
