# Ranking and visualization protocol

## Workflow

`main_v2.ipynb` runs in this order: (1) load matching cached runs or train missing/stale experiments; (2) evaluate all configured metrics and score each experiment; (3) display the four setting-specific final tables and metric summaries; and (4) create summary plots. `run_experiment` does not call plotting functions. The diagnostic smoke test is disabled by default.

The four settings are univariate SPY hourly, univariate SPY daily, multivariate hourly, and multivariate daily. Every final table reserves five model rows. A model without successful evaluation in a setting receives NA ranks for that setting.

## Project scoring and rank conversion

The teammate-proposed benchmark-normalized composite is computed per experiment. Each metric error is normalized against an iid Gaussian reference from training-only moments; predictive ratios are transformed with `abs(log(ratio))`. Cluster weights are distribution/tails 35%, temporal structure 35%, dependence 20%, and fidelity/utility 10%, with a 15% worst-cluster penalty. Lower scores are better. Univariate experiments omit dependence and portfolio VaR, then renormalize active cluster weights.

Three project ranking methods are converted to unique ordinal ranks within every experiment (1 = best):

- **Composite Rank:** ascending composite score. Exact ties are resolved by model name so every successfully evaluated model gets a distinct rank.
- **Pareto Rank:** first sort by nondominated front across available cluster scores; front 1 is best. Models on the same front are ordered by their unweighted mean active cluster score, then model name. `Pareto Front` remains in the per-experiment output to make the underlying partial order visible.
- **Lexicographic Rank:** ascending score tuple in the priority order distribution/tails, downstream predictive usefulness, temporal structure, multivariate dependence when applicable, then discriminative score. Exact ties are resolved by model name.

The final rank for each model and setting is computed in two stages: calculate the three project ranks separately for each training-duration experiment, then take the median rank across available durations. Sort the models by that median and convert it to a unique 1..N ordinal rank within the setting. Missing model-duration results are omitted from the median; no score is imputed. The median-rank columns are retained in `rankings_by_segment.csv` for transparency.

## Literature evidence

Published comparisons do not rank the same five exact implementations on a shared benchmark. `literature_reference_ranks.csv` retains source-specific reported or derived ranks and match caveats. The exploratory **Research Evidence Rank** applies Pareto comparisons over normalized source ranks, comparing a pair only on sources that evaluated both. It then orders within Pareto fronts by the mean normalized rank; source coverage is shown. This is a partial-overlap synthesis, not a validated universal leaderboard. A model with no comparative published result, including StableGARCH in the selected sources, remains NA; lack of evidence is not treated as poor performance.

The four setting tables include the same broad research-evidence ordering for context; only project-specific ranks differ by frequency and dimensionality. Do not interpret a literature rank as a controlled head-to-head comparison with this project's implementation.

## Plots

All plot calls appear after metric and ranking tables. The summary figures are:

- `final_rank_heatmaps.png`: research evidence and the three project ranks for all four settings.
- `method_rank_transitions.png`: model rank changes across the composite, Pareto, and lexicographic methods.
- `cluster_score_profiles.png`: median objective-cluster errors, showing why models rank differently.
- `pareto_tradeoffs.png`: distribution/tails versus downstream predictive-usefulness errors; stars mark the median front-1 models.
- `literature_evidence_map.png`: source-specific rank coverage and values, with NA where a source provides no result.

## Output files

All are written under `results_v2/`:

- `rankings_per_experiment.csv`: metric values and ordinal project ranks per duration experiment.
- `rankings_by_segment.csv`: median per-duration ranks and final ordinal ranks for all five models in all four settings.
- `final_rankings_by_setting.csv`: the final four-setting comparison table, with research evidence, project ranks, source-specific evidence, and coverage caveats.
- `literature_sources.csv`, `literature_reference_ranks.csv`, `literature_model_coverage.csv`, and `literature_evidence_pareto.csv`: source details and evidence coverage.
- Summary plot PNGs under `results_v2/plots/`.

The Gaussian reference is a normalization baseline, not a realism target or pass/fail threshold. All raw metric values remain in the experiment tables.
