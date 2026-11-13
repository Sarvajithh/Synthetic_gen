"""Build project score rankings and source-specific literature comparisons.

Literature ranks are kept source-specific: benchmark datasets, metric suites,
and candidate sets differ, so this module deliberately does not pool them into
an overall literature leaderboard.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_scoring import LEXICOGRAPHIC_ORDER


PROJECT_MODELS = ["TimeGAN", "TimeVAE", "DiffusionTS", "StudentTARIMAGARCH", "StableGARCH"]

LITERATURE_SOURCES = [
    {"source_id": "TSGBench_2024", "title": "TSGBench: Time Series Generation Benchmark",
     "url": "https://www.vldb.org/pvldb/vol17/p305-huang.pdf", "rank_status": "reported",
     "scope": "General time-series generation; 10 models, multiple datasets and measures",
     "limitations": "Not finance-specific; ranks are across the paper's benchmark."},
    {"source_id": "TSDiff_NeurIPS_2023", "title": "Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting",
     "url": "https://papers.nips.cc/paper_files/paper/2023/file/5a1a10c2c2c9b9af1514687bc24b8f3d-Paper-Conference.pdf",
     "rank_status": "derived", "scope": "Downstream forecasting CRPS on eight general time-series datasets",
     "limitations": "Derived paired ranks from Table 4; TSDiff is not this project's DiffusionTS."},
    {"source_id": "PaD_TS_AAAI_2025", "title": "Population Aware Diffusion for Time Series Generation",
     "url": "https://ojs.aaai.org/index.php/AAAI/article/download/34038/36193", "rank_status": "derived",
     "scope": "Google Stocks dataset; ranks from four Table 1 measures across four methods",
     "limitations": "Ranks calculated from reported values; includes PaD-TS in candidate set; not a finance-wide leaderboard."},
    {"source_id": "Financial_ARIMA_GARCH_arXiv_2025", "title": "Synthetic Financial Data Generation for Enhanced Financial Modelling",
     "url": "https://arxiv.org/abs/2512.21791", "rank_status": "derived",
     "scope": "S&P 500 daily log returns; MMD and KS results for TimeGAN, VAE, ARIMA-GARCH",
     "limitations": "Preprint; generic VAE and ARIMA-GARCH are family/nearest analogues, not exact project implementations."},
    {"source_id": "StableGARCH_1995", "title": "Stable GARCH models for financial time series",
     "url": "https://doi.org/10.1016/0893-9659(95)00063-V", "rank_status": "no_generator_rank",
     "scope": "Method paper for stable GARCH models",
     "limitations": "No synthetic-generator head-to-head ranking reported."},
]

# Reported average rank is retained when the source reports one. Derived values
# state their candidate set and basis; they must not be presented as paper claims.
LITERATURE_RANKS = [
    {"source_id": "TSGBench_2024", "model": "TimeVAE", "average_rank": 3.83, "comparison_set_size": 10,
     "rank_basis": "Reported average rank (Figure 8)", "match": "exact model name"},
    {"source_id": "TSGBench_2024", "model": "TimeGAN", "average_rank": 6.31, "comparison_set_size": 10,
     "rank_basis": "Reported average rank (Figure 8)", "match": "exact model name"},
    {"source_id": "TSDiff_NeurIPS_2023", "model": "TimeVAE", "average_rank": 1.3333333333, "comparison_set_size": 2,
     "rank_basis": "Derived: 16 wins in 24 paired lower-CRPS comparisons", "match": "exact model name"},
    {"source_id": "TSDiff_NeurIPS_2023", "model": "TimeGAN", "average_rank": 1.6666666667, "comparison_set_size": 2,
     "rank_basis": "Derived: 8 wins in 24 paired lower-CRPS comparisons", "match": "exact model name"},
    {"source_id": "PaD_TS_AAAI_2025", "model": "DiffusionTS", "average_rank": 2.625, "comparison_set_size": 4,
     "rank_basis": "Derived average rank over VDS, FDDS, DA, predictive score; lower is better", "match": "Diffusion-TS name/architecture match"},
    {"source_id": "PaD_TS_AAAI_2025", "model": "TimeGAN", "average_rank": 3.25, "comparison_set_size": 4,
     "rank_basis": "Derived average rank over VDS, FDDS, DA, predictive score; lower is better", "match": "exact model name"},
    {"source_id": "PaD_TS_AAAI_2025", "model": "TimeVAE", "average_rank": 3.0, "comparison_set_size": 4,
     "rank_basis": "Derived average rank over VDS, FDDS, DA, predictive score; lower is better", "match": "exact model name"},
    {"source_id": "Financial_ARIMA_GARCH_arXiv_2025", "model": "TimeGAN", "average_rank": 1.0, "comparison_set_size": 3,
     "rank_basis": "Derived average rank over reported MMD and KS values", "match": "exact model name"},
    {"source_id": "Financial_ARIMA_GARCH_arXiv_2025", "model": "TimeVAE", "average_rank": 2.0, "comparison_set_size": 3,
     "rank_basis": "Generic VAE comparator; derived average rank over MMD and KS", "match": "family analogue only"},
    {"source_id": "Financial_ARIMA_GARCH_arXiv_2025", "model": "StudentTARIMAGARCH", "average_rank": 3.0, "comparison_set_size": 3,
     "rank_basis": "ARIMA-GARCH comparator; derived average rank over MMD and KS", "match": "nearest analogue only"},
]

MODEL_COVERAGE = [
    {"model": "TimeGAN", "coverage": "Ranks in TSGBench, TSDiff, PaD-TS, and financial preprint", "caveat": "Different tasks, datasets, metrics, and candidate sets."},
    {"model": "TimeVAE", "coverage": "Ranks in TSGBench, TSDiff, and PaD-TS; generic VAE analogue in financial preprint", "caveat": "Financial-preprint VAE is not identified as TimeVAE."},
    {"model": "DiffusionTS", "coverage": "PaD-TS paper reports Diffusion-TS comparator values", "caveat": "Architecture/name match; implementation and protocol need not match."},
    {"model": "StudentTARIMAGARCH", "coverage": "ARIMA-GARCH nearest-analogue rank in financial preprint", "caveat": "Not an exact Student-t model or necessarily matching ARIMA/GARCH orders."},
    {"model": "StableGARCH", "coverage": "No synthetic-generator comparative rank found", "caveat": "Stable GARCH method paper is not a generator leaderboard."},
]

SCORE_ALIASES = {
    "Friend Composite Score": "Benchmark-Normalized Composite Score",
    "Composite Error Score": "Benchmark-Normalized Composite Score",
    "Friend Distribution Tails Score": "Distribution/Tails Cluster Score",
    "Friend Temporal Structure Score": "Temporal Structure Cluster Score",
    "Friend Multivariate Dependence Score": "Multivariate Dependence Cluster Score",
    "Friend Fidelity And Utility Score": "Fidelity/Utility Cluster Score",
}


def pareto_fronts(frame: pd.DataFrame, objective_columns: list[str]) -> pd.Series:
    """Return nondominated sorting fronts; all objective columns are minimized."""
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    valid = frame[objective_columns].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    remaining = list(frame.index[valid])
    front = 1
    while remaining:
        current = []
        for idx in remaining:
            candidate = frame.loc[idx, objective_columns].to_numpy(dtype=float)
            if not any(other_idx != idx and
                       np.all(frame.loc[other_idx, objective_columns].to_numpy(dtype=float) <= candidate) and
                       np.any(frame.loc[other_idx, objective_columns].to_numpy(dtype=float) < candidate)
                       for other_idx in remaining):
                current.append(idx)
        result.loc[current] = front
        remaining = [idx for idx in remaining if idx not in set(current)]
        front += 1
    return result


def _normalize_legacy_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for old, new in SCORE_ALIASES.items():
        if old in out:
            if new not in out:
                out[new] = out[old]
            else:
                out[new] = out[new].where(out[new].notna(), out[old])
    return out


def rank_experiment(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach composite, Pareto-front and lexicographic ranks per experiment."""
    out = _normalize_legacy_columns(frame)
    status = out.get("Model Status", pd.Series("ok", index=out.index)).astype(str).str.lower()
    successful = status.eq("ok")
    score_col = "Benchmark-Normalized Composite Score"
    out["Composite Rank"] = np.nan
    if score_col in out:
        out.loc[successful, "Composite Rank"] = pd.to_numeric(
            out.loc[successful, score_col], errors="coerce").rank(method="min", ascending=True)

    objectives = [c for c in ["Distribution/Tails Cluster Score", "Temporal Structure Cluster Score",
                               "Fidelity/Utility Cluster Score"] if c in out]
    dep = "Multivariate Dependence Cluster Score"
    if dep in out and out[dep].notna().any():
        objectives.insert(2, dep)
    out["Pareto Front"] = np.nan
    if objectives:
        eligible = out.loc[successful]
        out.loc[successful, "Pareto Front"] = pareto_fronts(eligible, objectives)

    lex_cols = [f"Lexicographic {name.replace('_', ' ').title()}" for name in LEXICOGRAPHIC_ORDER]
    lex_cols = [c for c in lex_cols if c in out and out[c].notna().any()]
    out["Lexicographic Rank"] = np.nan
    if lex_cols:
        eligible = out.loc[successful & out[lex_cols].notna().all(axis=1)].copy()
        if not eligible.empty:
            eligible = eligible.sort_values(lex_cols + (["Model"] if "Model" in out else []), kind="mergesort")
            keys = eligible[lex_cols].apply(tuple, axis=1)
            key_to_rank = {key: rank for rank, key in enumerate(sorted(set(keys)), 1)}
            out.loc[eligible.index, "Lexicographic Rank"] = keys.map(key_to_rank)
    out["Ranking Status"] = np.where(successful, "ranked", "model_failed")
    return out


def _experiment_keys(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    out = frame.copy()
    if len(path.parts) >= 5:
        for col, value in [("Data Configuration", path.parts[-4]),
                           ("Frequency", path.parts[-3]), ("Training Duration", path.parts[-2])]:
            if col not in out:
                out[col] = value
    return out


def _literature_tables(output_dir: Path):
    sources = pd.DataFrame(LITERATURE_SOURCES)
    ranks = pd.DataFrame(LITERATURE_RANKS).merge(sources[["source_id", "title", "url", "scope", "limitations"]],
                                                 on="source_id", how="left")
    ranks.to_csv(output_dir / "literature_reference_ranks.csv", index=False)
    sources.to_csv(output_dir / "literature_sources.csv", index=False)
    coverage = pd.DataFrame(MODEL_COVERAGE)
    coverage.to_csv(output_dir / "literature_model_coverage.csv", index=False)
    # This file intentionally contains no pooled ranks; cross-paper pooling is
    # invalid because papers use different protocols and benchmark candidate sets.
    pd.DataFrame([{"status": "not_computed", "reason": "Source ranks are not pooled across incompatible datasets, metrics, and candidate sets. Use literature_reference_ranks.csv."}]
                 ).to_csv(output_dir / "literature_aggregate_ranks.csv", index=False)
    return sources, ranks, coverage


def write_ranking_outputs(results_dir: str | Path) -> dict[str, pd.DataFrame]:
    results_dir = Path(results_dir)
    files = sorted(p for p in results_dir.rglob("metrics.csv") if p.parent != results_dir)
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        if "Model" in frame:
            frames.append(rank_experiment(_experiment_keys(frame, path)))
    if not frames:
        raise FileNotFoundError(f"No per-experiment metrics.csv files found under {results_dir}")
    per_exp = pd.concat(frames, ignore_index=True)
    per_exp.to_csv(results_dir / "rankings_per_experiment.csv", index=False)
    rank_cols = ["Composite Rank", "Pareto Front", "Lexicographic Rank"]
    valid = per_exp[per_exp["Ranking Status"].eq("ranked")]
    segment = (valid.groupby(["Data Configuration", "Frequency", "Model"], as_index=False)
               .agg(**{f"Mean {c}": (c, "mean") for c in rank_cols},
                    Experiments=("Training Duration", "nunique")))
    for col, label in zip(rank_cols, ["Composite", "Pareto", "Lexicographic"]):
        segment[f"{label} Segment Rank"] = segment.groupby(["Data Configuration", "Frequency"])[f"Mean {col}"].rank(method="min")
    segment.to_csv(results_dir / "rankings_by_segment.csv", index=False)
    sources, ref_ranks, coverage = _literature_tables(results_dir)

    # Keep all five project models in the comparison. Each literature source
    # has its own rank column; blanks mean no published compatible rank found.
    project = per_exp.copy()
    for source in ref_ranks["source_id"].unique():
        source_rows = ref_ranks[ref_ranks.source_id.eq(source)].set_index("model")["average_rank"]
        project[f"Literature Rank | {source}"] = project["Model"].map(source_rows)
    project = project.merge(coverage, left_on="Model", right_on="model", how="left").drop(columns=["model"])
    project.to_csv(results_dir / "scores_and_rankings.csv", index=False)
    project.to_csv(results_dir / "project_vs_literature.csv", index=False)
    return {"per_experiment": per_exp, "by_segment": segment, "literature_sources": sources,
            "literature_reference_ranks": ref_ranks, "literature_model_coverage": coverage,
            "literature_aggregate": pd.DataFrame([{"status": "not_computed"}]),
            "project_vs_literature": project}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results_v2")
    args = parser.parse_args()
    for name, frame in write_ranking_outputs(args.results_dir).items():
        print(f"{name}: {len(frame)} rows")


if __name__ == "__main__":
    main()
