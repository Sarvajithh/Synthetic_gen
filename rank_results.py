"""Build ordinal project rankings and a partial literature-evidence ranking."""

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


def _ordinal_ranks(frame: pd.DataFrame, sort_columns: list[str], ascending: list[bool] | None = None) -> pd.Series:
    """Assign unique 1..N ranks; stable model-name ordering breaks exact ties."""
    if frame.empty:
        return pd.Series(dtype=float, index=frame.index)
    sort_columns = [c for c in sort_columns if c in frame]
    model_col = "Model" if "Model" in frame else ("model" if "model" in frame else None)
    tie_columns = [model_col] if model_col and model_col not in sort_columns else []
    ordered = frame.sort_values(sort_columns + tie_columns,
                                ascending=(ascending or [True] * len(sort_columns)) +
                                ([True] * len(tie_columns)),
                                kind="mergesort", na_position="last")
    ranks = pd.Series(np.arange(1, len(ordered) + 1, dtype=int), index=ordered.index)
    return ranks.reindex(frame.index)


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
        eligible = out.loc[successful & pd.to_numeric(out[score_col], errors="coerce").notna()].copy()
        eligible[score_col] = pd.to_numeric(eligible[score_col], errors="coerce")
        out.loc[eligible.index, "Composite Rank"] = _ordinal_ranks(eligible, [score_col])

    objectives = [c for c in ["Distribution/Tails Cluster Score", "Temporal Structure Cluster Score",
                               "Fidelity/Utility Cluster Score"] if c in out]
    dep = "Multivariate Dependence Cluster Score"
    if dep in out and out[dep].notna().any():
        objectives.insert(2, dep)
    out["Pareto Front"] = np.nan
    out["Pareto Rank"] = np.nan
    if objectives:
        eligible = out.loc[successful]
        out.loc[successful, "Pareto Front"] = pareto_fronts(eligible, objectives)
        eligible = eligible.copy()
        eligible["Pareto Front"] = out.loc[eligible.index, "Pareto Front"]
        eligible["Pareto Tie Break"] = eligible[objectives].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        eligible = eligible[eligible["Pareto Front"].notna()]
        out.loc[eligible.index, "Pareto Rank"] = _ordinal_ranks(
            eligible, ["Pareto Front", "Pareto Tie Break"])

    lex_cols = [f"Lexicographic {name.replace('_', ' ').title()}" for name in LEXICOGRAPHIC_ORDER]
    lex_cols = [c for c in lex_cols if c in out and out[c].notna().any()]
    out["Lexicographic Rank"] = np.nan
    if lex_cols:
        eligible = out.loc[successful & out[lex_cols].notna().all(axis=1)].copy()
        if not eligible.empty:
            out.loc[eligible.index, "Lexicographic Rank"] = _ordinal_ranks(eligible, lex_cols)
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


def _literature_evidence_ranks(reference_ranks: pd.DataFrame) -> pd.DataFrame:
    """Combine source ranks by Pareto dominance on sources shared by each pair.

    This is an exploratory literature-evidence ordering, not a universal
    leaderboard. Models with no published comparator remain unranked (NA).
    """
    ranks = reference_ranks.copy()
    ranks["Normalized Source Rank"] = ((ranks["average_rank"] - 1) /
                                        (ranks["comparison_set_size"] - 1)).clip(0, 1)
    matrix = ranks.pivot_table(index="model", columns="source_id", values="Normalized Source Rank", aggfunc="mean")
    names = list(matrix.index)
    remaining = names.copy()
    fronts = {}
    front = 1
    while remaining:
        current = []
        for candidate in remaining:
            candidate_values = matrix.loc[candidate]
            is_dominated = False
            for other in remaining:
                if other == candidate:
                    continue
                other_values = matrix.loc[other]
                shared = candidate_values.notna() & other_values.notna()
                if shared.any() and (other_values[shared] <= candidate_values[shared]).all() and \
                        (other_values[shared] < candidate_values[shared]).any():
                    is_dominated = True
                    break
            if not is_dominated:
                current.append(candidate)
        for name in current:
            fronts[name] = front
        if not current:  # defensive guard for non-transitive partial comparisons
            for name in remaining:
                fronts[name] = front
            break
        remaining = [name for name in remaining if name not in set(current)]
        front += 1

    evidence = matrix.copy()
    evidence["Research Evidence Pareto Front"] = pd.Series(fronts)
    evidence["Research Evidence Rank"] = np.nan
    evidence["Literature Sources Count"] = matrix.notna().sum(axis=1)
    # Pareto fronts are the primary order. Within a front, normalized mean
    # rank gives a reproducible ordinal tie-break; source coverage is reported.
    evidence["Mean Normalized Literature Rank"] = matrix.mean(axis=1)
    eligible = evidence.reset_index()
    eligible["Research Evidence Rank"] = _ordinal_ranks(
        eligible, ["Research Evidence Pareto Front", "Mean Normalized Literature Rank"])
    evidence["Research Evidence Rank"] = eligible.set_index("model")["Research Evidence Rank"]
    return evidence.reset_index()


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
    rank_cols = ["Composite Rank", "Pareto Rank", "Lexicographic Rank"]
    valid = per_exp[per_exp["Ranking Status"].eq("ranked")]
    segment = (valid.groupby(["Data Configuration", "Frequency", "Model"], as_index=False)
               .agg(**{f"Median {c}": (c, "median") for c in rank_cols},
                    Experiments=("Training Duration", "nunique")))
    setting_frame = pd.DataFrame([
        {"Data Configuration": data_config, "Frequency": frequency, "Model": model}
        for data_config in ["univariate_spy", "multivariate"]
        for frequency in ["hourly", "daily"]
        for model in PROJECT_MODELS
    ])
    segment = setting_frame.merge(segment, on=["Data Configuration", "Frequency", "Model"], how="left")
    segment["Experiments"] = segment["Experiments"].fillna(0).astype(int)
    # Convert median ranks over durations back to unique 1..5 ordinals per
    # setting and method, with model name as a deterministic tie-breaker.
    for col, label in zip(rank_cols, ["Composite", "Pareto", "Lexicographic"]):
        segment[f"{label} Rank"] = np.nan
        for _, indices in segment.groupby(["Data Configuration", "Frequency"]).groups.items():
            subset = segment.loc[indices]
            eligible = subset[subset[f"Median {col}"].notna()]
            if not eligible.empty:
                segment.loc[eligible.index, f"{label} Rank"] = _ordinal_ranks(eligible, [f"Median {col}"])
    segment.to_csv(results_dir / "rankings_by_segment.csv", index=False)
    sources, ref_ranks, coverage = _literature_tables(results_dir)

    evidence_ranks = _literature_evidence_ranks(ref_ranks)
    evidence_ranks.to_csv(results_dir / "literature_evidence_pareto.csv", index=False)
    # Final comparison table: four combinations, one row per project generator.
    project = segment.copy()
    for source in ref_ranks["source_id"].unique():
        source_rows = ref_ranks[ref_ranks.source_id.eq(source)].set_index("model")["average_rank"]
        project[f"Literature Rank | {source}"] = project["Model"].map(source_rows)
    project = project.merge(evidence_ranks[["model", "Research Evidence Pareto Front", "Research Evidence Rank",
                                            "Literature Sources Count", "Mean Normalized Literature Rank"]],
                            left_on="Model", right_on="model", how="left").drop(columns=["model"])
    project = project.merge(coverage, left_on="Model", right_on="model", how="left").drop(columns=["model"])
    project["Setting"] = project["Data Configuration"] + " / " + project["Frequency"]
    project["Literature Rank | StableGARCH_1995"] = np.nan
    table_columns = ["Setting", "Data Configuration", "Frequency", "Model", "Research Evidence Rank",
                     "Composite Rank", "Pareto Rank", "Lexicographic Rank",
                     "Research Evidence Pareto Front", "Literature Sources Count", "coverage", "caveat"]
    table_columns += [f"Literature Rank | {source}" for source in ref_ranks["source_id"].unique()]
    project = project[[c for c in table_columns if c in project.columns] +
                      [c for c in project.columns if c not in table_columns]]
    # Sort each setting by research-evidence rank. Uncovered models remain in
    # the table with NA and are not declared worst performers.
    project = project.sort_values(["Data Configuration", "Frequency", "Research Evidence Rank", "Model"],
                                  kind="mergesort", na_position="last")
    project.to_csv(results_dir / "final_rankings_by_setting.csv", index=False)
    project.to_csv(results_dir / "scores_and_rankings.csv", index=False)
    project.to_csv(results_dir / "project_vs_literature.csv", index=False)
    return {"per_experiment": per_exp, "by_segment": segment, "literature_sources": sources,
            "literature_reference_ranks": ref_ranks, "literature_model_coverage": coverage,
            "literature_evidence_pareto": evidence_ranks, "final_rankings": project,
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
