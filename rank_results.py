"""Build per-experiment, segment, and literature-comparison ranking tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_scoring import LEXICOGRAPHIC_ORDER


LITERATURE_SOURCES = [
    {
        "source_id": "TSGBench_2024",
        "title": "TSGBench: Time Series Generation Benchmark",
        "url": "https://www.vldb.org/pvldb/vol17/p305-huang.pdf",
        "scope": "General time-series generation; 10 methods, multiple datasets and measures",
        "contributes_to_aggregate": True,
        "limitations": "Not finance-specific; project model labels are architecture-level matches.",
    },
    {
        "source_id": "TSDiff_NeurIPS_2023",
        "title": "Unconditional Time Series Generation with Diffusion Models (TSDiff)",
        "url": "https://papers.nips.cc/paper_files/paper/2023/file/5a1a10c2c2c9b9af1514687bc24b8f3d-Paper-Conference.pdf",
        "scope": "Downstream forecasting utility on eight general time-series datasets",
        "contributes_to_aggregate": True,
        "limitations": "Derived TimeVAE/TimeGAN paired ranks from Table 4; not finance-specific; TSDiff is not this project's DiffusionTS.",
    },
    {
        "source_id": "FTS_Diffusion_ICLR_2024",
        "title": "Generative Learning for Financial Time Series with Irregular and Scale-Invariant Patterns",
        "url": "https://proceedings.iclr.cc/paper_files/paper/2024/file/f90fc76b199fe6b0ec2a51aaf72c3277-Paper-Conference.pdf",
        "scope": "Financial return distribution and downstream prediction comparisons",
        "contributes_to_aggregate": False,
        "limitations": "Only TimeGAN overlaps by model identity; one common model cannot establish a rank comparison.",
    },
    {
        "source_id": "Financial_Diffusion_2025",
        "title": "Generation of synthetic financial time series by diffusion models",
        "url": "https://www.tandfonline.com/doi/full/10.1080/14697688.2025.2528697",
        "scope": "Finance-specific comparison of diffusion variants, TimeGAN and QuantGAN",
        "contributes_to_aggregate": False,
        "limitations": "Only TimeGAN is a direct model-name overlap; diffusion implementations are not identical.",
    },
]

# TSGBench Figure 8 reports these average ranks across its 10-method benchmark.
# The second paper's paired ranks are derived from the 24 (8 datasets x 3
# downstream forecasters) TimeVAE-vs-TimeGAN CRPS comparisons in its Table 4.
LITERATURE_RANKS = [
    {"source_id": "TSGBench_2024", "model": "TimeVAE", "average_rank": 3.83,
     "comparison_set_size": 10, "rank_basis": "Reported average rank across benchmark measures/datasets"},
    {"source_id": "TSGBench_2024", "model": "TimeGAN", "average_rank": 6.31,
     "comparison_set_size": 10, "rank_basis": "Reported average rank across benchmark measures/datasets"},
    {"source_id": "TSDiff_NeurIPS_2023", "model": "TimeVAE", "average_rank": 1.3333333333,
     "comparison_set_size": 2, "rank_basis": "Derived: 16 first-place results in 24 paired lower-CRPS comparisons"},
    {"source_id": "TSDiff_NeurIPS_2023", "model": "TimeGAN", "average_rank": 1.6666666667,
     "comparison_set_size": 2, "rank_basis": "Derived: 8 first-place results in 24 paired lower-CRPS comparisons"},
]


def pareto_fronts(frame: pd.DataFrame, objective_columns: list[str]) -> pd.Series:
    """Return nondominated sorting fronts; objectives are all minimized."""
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    valid = frame[objective_columns].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    remaining = list(frame.index[valid])
    front = 1
    while remaining:
        current = []
        for idx in remaining:
            candidate = frame.loc[idx, objective_columns].to_numpy(dtype=float)
            dominated = False
            for other_idx in remaining:
                if idx == other_idx:
                    continue
                other = frame.loc[other_idx, objective_columns].to_numpy(dtype=float)
                if np.all(other <= candidate) and np.any(other < candidate):
                    dominated = True
                    break
            if not dominated:
                current.append(idx)
        result.loc[current] = front
        remaining = [idx for idx in remaining if idx not in set(current)]
        front += 1
    return result


def rank_experiment(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach composite, Pareto-front, and lexicographic ranks to one experiment."""
    out = frame.copy()
    status = out.get("Model Status", pd.Series("ok", index=out.index)).astype(str).str.lower()
    successful = status.eq("ok")
    score_col = "Friend Composite Score" if "Friend Composite Score" in out else "Composite Error Score"
    out["Friend Rank"] = np.nan
    out.loc[successful, "Friend Rank"] = out.loc[successful, score_col].rank(method="min", ascending=True)

    objective_cols = [f"Friend {name.replace('_', ' ').title()} Score"
                      for name in ("distribution_tails", "temporal_structure",
                                   "fidelity_and_utility")]
    dependence_col = "Friend Multivariate Dependence Score"
    if dependence_col in out and out[dependence_col].notna().any():
        objective_cols.insert(2, dependence_col)
    objective_cols = [c for c in objective_cols if c in out]
    out["Pareto Front"] = np.nan
    if objective_cols:
        eligible = out.loc[successful]
        out.loc[successful, "Pareto Front"] = pareto_fronts(eligible, objective_cols)

    lex_columns = []
    for objective in LEXICOGRAPHIC_ORDER:
        col = f"Lexicographic {objective.replace('_', ' ').title()}"
        if col in out and out[col].notna().any():
            lex_columns.append(col)
    out["Lexicographic Rank"] = np.nan
    if lex_columns:
        eligible = out.loc[successful & out[lex_columns].notna().all(axis=1)].copy()
        if len(eligible):
            eligible = eligible.sort_values(lex_columns + ["Model"], kind="mergesort")
            keys = eligible[lex_columns].apply(tuple, axis=1)
            ordered_keys = sorted(set(keys))
            key_to_rank = {key: rank for rank, key in enumerate(ordered_keys, start=1)}
            eligible["Lexicographic Rank"] = keys.map(key_to_rank)
            out.loc[eligible.index, "Lexicographic Rank"] = eligible["Lexicographic Rank"]

    out["Ranking Status"] = np.where(successful, "ranked", "model_failed")
    return out


def _add_experiment_keys(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    out = frame.copy()
    parts = path.parts
    # results_dir / data_config / frequency / duration / metrics.csv
    if len(parts) >= 5:
        out["Data Configuration"] = out.get("Data Configuration", parts[-4])
        out["Frequency"] = out.get("Frequency", parts[-3])
        out["Training Duration"] = out.get("Training Duration", parts[-2])
    return out


def _literature_tables(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sources = pd.DataFrame(LITERATURE_SOURCES)
    ranks = pd.DataFrame(LITERATURE_RANKS).merge(
        sources[["source_id", "title", "url", "scope", "contributes_to_aggregate", "limitations"]],
        on="source_id", how="left")
    # Convert rank to [0,1], where 0 is first. Normalize for field size before
    # averaging across sources; this remains a descriptive, heterogeneous pool.
    ranks["normalized_rank_percentile"] = (
        (ranks["average_rank"] - 1) / (ranks["comparison_set_size"] - 1)
    ).clip(0, 1)
    included = ranks[ranks["contributes_to_aggregate"]].copy()
    aggregate = (included.groupby("model", as_index=False)
                 .agg(mean_normalized_literature_rank=("normalized_rank_percentile", "mean"),
                      literature_sources=("source_id", "nunique"),
                      source_rank_values=("average_rank", lambda s: "; ".join(f"{x:.3f}" for x in s))))
    aggregate["literature_aggregate_rank"] = aggregate["mean_normalized_literature_rank"].rank(
        method="min", ascending=True)
    aggregate["interpretation"] = (
        "Exploratory equal-source mean of normalized ranks; papers use different datasets, metrics, and candidate sets."
    )
    sources.to_csv(output_dir / "literature_sources.csv", index=False)
    ranks.to_csv(output_dir / "literature_reference_ranks.csv", index=False)
    aggregate.to_csv(output_dir / "literature_aggregate_ranks.csv", index=False)
    return ranks, aggregate


def write_ranking_outputs(results_dir: str | Path) -> dict[str, pd.DataFrame]:
    results_dir = Path(results_dir)
    files = sorted(results_dir.rglob("metrics.csv"))
    frames = []
    for path in files:
        if path.parent == results_dir:
            continue
        frame = pd.read_csv(path)
        if "Model" not in frame.columns:
            continue
        frame = _add_experiment_keys(frame, path)
        frames.append(rank_experiment(frame))

    if not frames:
        raise FileNotFoundError(f"No per-experiment metrics.csv files found under {results_dir}")

    per_experiment = pd.concat(frames, ignore_index=True)
    per_experiment.to_csv(results_dir / "rankings_per_experiment.csv", index=False)

    # Mean rank across training-duration experiments, keeping dataset and
    # sampling frequency separate. Failed runs are absent from rank means.
    rank_columns = ["Friend Rank", "Pareto Front", "Lexicographic Rank"]
    valid = per_experiment[per_experiment["Ranking Status"] == "ranked"]
    segment = (valid.groupby(["Data Configuration", "Frequency", "Model"], as_index=False)
               .agg(**{f"Mean {col}": (col, "mean") for col in rank_columns},
                    Experiments=("Training Duration", "nunique")))
    for col, label in zip(rank_columns, ["Friend", "Pareto", "Lexicographic"]):
        segment[f"{label} Segment Rank"] = segment.groupby(
            ["Data Configuration", "Frequency"])[f"Mean {col}"].rank(method="min", ascending=True)
    segment.to_csv(results_dir / "rankings_by_segment.csv", index=False)

    literature_ranks, literature_aggregate = _literature_tables(results_dir)
    common = segment.merge(literature_aggregate, left_on="Model", right_on="model", how="inner")
    common.to_csv(results_dir / "project_vs_literature.csv", index=False)
    return {
        "per_experiment": per_experiment,
        "by_segment": segment,
        "literature_sources": pd.DataFrame(LITERATURE_SOURCES),
        "literature_reference_ranks": literature_ranks,
        "literature_aggregate": literature_aggregate,
        "project_vs_literature": common,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results_v2")
    args = parser.parse_args()
    outputs = write_ranking_outputs(args.results_dir)
    for name, frame in outputs.items():
        print(f"{name}: {len(frame)} rows")


if __name__ == "__main__":
    main()
