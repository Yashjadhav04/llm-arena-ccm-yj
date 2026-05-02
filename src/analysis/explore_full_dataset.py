from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.arena_dataset import flatten_initial_features, load_arena_raw


PERCENTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]


def _markdown_table(frame: pd.DataFrame, index: bool = False) -> list[str]:
    render = frame.copy()
    if not index:
        render = render.reset_index(drop=True)

    columns = list(render.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, separator]
    for row in render.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _format_number(value: float | int | str) -> str:
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "nan"
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:.4f}"
    return str(value)


def _top_counts(series: pd.Series, top_n: int = 10, normalize: bool = False) -> pd.DataFrame:
    counts = series.fillna("missing").astype(str).value_counts(dropna=False, normalize=normalize)
    output = counts.head(top_n).rename("share" if normalize else "count").reset_index()
    output.columns = ["value", "share" if normalize else "count"]
    return output


def _numeric_summary(frame: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in numeric_columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        summary = series.describe(percentiles=PERCENTILES)
        rows.append(
            {
                "feature": column,
                "non_null": int(summary["count"]),
                "missing": int(series.isna().sum()),
                "mean": float(summary["mean"]) if summary["count"] else np.nan,
                "std": float(summary["std"]) if summary["count"] else np.nan,
                "min": float(summary["min"]) if summary["count"] else np.nan,
                "p01": float(series.quantile(0.01)) if summary["count"] else np.nan,
                "p05": float(series.quantile(0.05)) if summary["count"] else np.nan,
                "p25": float(summary["25%"]) if summary["count"] else np.nan,
                "median": float(summary["50%"]) if summary["count"] else np.nan,
                "p75": float(summary["75%"]) if summary["count"] else np.nan,
                "p95": float(series.quantile(0.95)) if summary["count"] else np.nan,
                "p99": float(series.quantile(0.99)) if summary["count"] else np.nan,
                "max": float(summary["max"]) if summary["count"] else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _boolean_summary(frame: pd.DataFrame, bool_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = len(frame)
    for column in bool_columns:
        series = frame[column]
        true_count = int(series.fillna(False).astype(bool).sum())
        missing = int(series.isna().sum())
        false_count = int(total - true_count - missing)
        rows.append(
            {
                "feature": column,
                "true_count": true_count,
                "true_share": true_count / total if total else np.nan,
                "false_count": false_count,
                "missing": missing,
            }
        )
    return pd.DataFrame(rows).sort_values("true_count", ascending=False, ignore_index=True)


def _feature_cardinality(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in frame.columns:
        series = frame[column]
        rows.append(
            {
                "feature": column,
                "dtype": str(series.dtype),
                "non_null": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "unique_non_null": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values("feature", ignore_index=True)


def _combined_model_frequency(flat: pd.DataFrame) -> pd.DataFrame:
    counts = pd.concat([flat["model_a"], flat["model_b"]], ignore_index=True).value_counts()
    output = counts.rename("appearances").reset_index()
    output.columns = ["model", "appearances"]
    output["appearance_share"] = output["appearances"] / (2 * len(flat))
    return output


def _session_summary(flat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    session_sizes = (
        flat.groupby("evaluation_session_id", dropna=False)
        .size()
        .rename("rows_in_session")
        .reset_index()
    )
    summary = pd.DataFrame(
        [
            {
                "sessions": int(session_sizes["evaluation_session_id"].nunique(dropna=True)),
                "mean_rows_per_session": float(session_sizes["rows_in_session"].mean()),
                "median_rows_per_session": float(session_sizes["rows_in_session"].median()),
                "p95_rows_per_session": float(session_sizes["rows_in_session"].quantile(0.95)),
                "max_rows_per_session": int(session_sizes["rows_in_session"].max()),
            }
        ]
    )
    size_distribution = (
        session_sizes["rows_in_session"].value_counts().sort_index().rename("session_count").reset_index()
    )
    size_distribution.columns = ["rows_in_session", "session_count"]
    size_distribution["share_of_sessions"] = (
        size_distribution["session_count"] / size_distribution["session_count"].sum()
    )
    return summary, size_distribution


def _length_bucket_counts(flat: pd.DataFrame, column: str) -> pd.DataFrame:
    bins = [-np.inf, 64, 128, 256, 512, 1024, np.inf]
    labels = ["<=64", "65-128", "129-256", "257-512", "513-1024", "1025+"]
    bucketed = pd.cut(flat[column], bins=bins, labels=labels)
    counts = bucketed.value_counts(dropna=False).rename("count").reset_index()
    counts.columns = ["bucket", "count"]
    counts["share"] = counts["count"] / counts["count"].sum()
    return counts


def _winner_by_group(flat: pd.DataFrame, group_col: str) -> pd.DataFrame:
    table = pd.crosstab(flat[group_col], flat["winner"], normalize="index")
    return table.reset_index()


def _length_outcome_tables(flat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_winner = (
        flat.groupby("winner", dropna=False)["length_diff_tokens"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )

    binary = flat.loc[flat["winner"].isin(["model_a", "model_b"])].copy()
    binary["a_longer"] = np.where(
        binary["length_diff_tokens"] > 0,
        "A longer",
        np.where(binary["length_diff_tokens"] < 0, "B longer", "same length"),
    )
    winner_given_longer = pd.crosstab(binary["a_longer"], binary["winner"], normalize="index")
    return by_winner, winner_given_longer.reset_index()


def build_markdown_report(
    source: str,
    flat: pd.DataFrame,
    feature_cardinality: pd.DataFrame,
    numeric_summary: pd.DataFrame,
    boolean_summary: pd.DataFrame,
    model_frequency: pd.DataFrame,
    session_summary: pd.DataFrame,
    session_size_distribution: pd.DataFrame,
    winner_counts: pd.DataFrame,
    language_counts: pd.DataFrame,
    task_bucket_counts: pd.DataFrame,
    evaluation_order_counts: pd.DataFrame,
    assistant_a_length_buckets: pd.DataFrame,
    assistant_b_length_buckets: pd.DataFrame,
    winner_by_task: pd.DataFrame,
    winner_by_code: pd.DataFrame,
    length_diff_by_winner: pd.DataFrame,
    binary_winner_given_longer: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# Full Dataset Exploratory Data Analysis")
    lines.append("")
    lines.append("## Dataset Overview")
    lines.append(f"- Data source: {source}")
    lines.append(f"- Rows: {len(flat):,}")
    lines.append(f"- Unique vote ids: {flat['id'].nunique():,}")
    lines.append(f"- Duplicate vote ids: {len(flat) - flat['id'].nunique():,}")
    lines.append(f"- Unique models across both sides: {len(model_frequency):,}")
    lines.append(f"- Unique evaluation sessions: {flat['evaluation_session_id'].nunique():,}")
    lines.append(f"- Binary votes (`model_a` or `model_b` wins): {int(flat['is_binary_vote'].sum()):,}")
    lines.append(f"- Non-binary votes (`tie` or `both_bad`): {int((~flat['is_binary_vote']).sum()):,}")
    lines.append(f"- Time range: {flat['timestamp'].min()} to {flat['timestamp'].max()}")
    lines.append("")
    lines.append("## Feature Cardinality")
    lines.extend(_markdown_table(feature_cardinality.head(20)))
    lines.append("")
    lines.append("The full cardinality table is saved to `results/eda_feature_cardinality.csv`.")
    lines.append("")
    lines.append("## Outcome Distribution")
    lines.extend(_markdown_table(winner_counts))
    lines.append("")
    lines.append("## Task Bucket Distribution")
    lines.extend(_markdown_table(task_bucket_counts))
    lines.append("")
    lines.append("## Language Distribution")
    lines.extend(_markdown_table(language_counts.head(15)))
    lines.append("")
    lines.append("## Boolean Feature Prevalence")
    lines.extend(_markdown_table(boolean_summary))
    lines.append("")
    lines.append("## Numeric Feature Summary")
    lines.extend(_markdown_table(numeric_summary))
    lines.append("")
    lines.append("## Session Structure")
    lines.extend(_markdown_table(session_summary))
    lines.append("")
    lines.append("### Session Size Distribution")
    lines.extend(_markdown_table(session_size_distribution.head(15)))
    lines.append("")
    lines.append("## Evaluation Order Distribution")
    lines.extend(_markdown_table(evaluation_order_counts.head(15)))
    lines.append("")
    lines.append("## Response Length Buckets")
    lines.append("### Assistant A Tokens")
    lines.extend(_markdown_table(assistant_a_length_buckets))
    lines.append("")
    lines.append("### Assistant B Tokens")
    lines.extend(_markdown_table(assistant_b_length_buckets))
    lines.append("")
    lines.append("## Model Frequency")
    lines.extend(_markdown_table(model_frequency.head(20)))
    lines.append("")
    lines.append("## Winner Distribution By Task Bucket")
    lines.extend(_markdown_table(winner_by_task))
    lines.append("")
    lines.append("## Winner Distribution By `is_code`")
    lines.extend(_markdown_table(winner_by_code))
    lines.append("")
    lines.append("## Length And Outcome")
    lines.append("### Length Difference By Winner")
    lines.extend(_markdown_table(length_diff_by_winner))
    lines.append("")
    lines.append("### Binary Winner Share Given Which Side Is Longer")
    lines.extend(_markdown_table(binary_winner_given_longer))
    lines.append("")
    lines.append("## Initial EDA Takeaways")
    lines.append("- The dataset includes both binary preference votes and a substantial number of `tie` / `both_bad` outcomes, so multinomial handling will matter later.")
    lines.append("- The task buckets are not balanced, with `mixed` and `factual_reasoning` dominating the first-pass split.")
    lines.append("- Language is diverse enough that an English-only sensitivity analysis is worth doing before interpreting global coefficients.")
    lines.append("- Response-length variables have wide tails, so robust scaling or log transforms are a good idea for modeling.")
    lines.append("- Longer responses are descriptively associated with winning pairwise votes, which supports testing verbosity bias directly in the first cognitive model.")
    lines.append("- Session and evaluation-order summaries can help decide whether to include session-level effects or clustered standard errors.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run descriptive EDA on the full Arena dataset.")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root containing data/, src/, docs/, and results/ directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/full data eda",
        help="Directory for EDA report and supporting CSV tables.",
    )
    parser.add_argument(
        "--output-markdown",
        default=None,
        help="Optional markdown report path. Defaults to output-dir/full_dataset_eda.md.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row cap for a smoke test. Default uses the full dataset.",
    )
    parser.add_argument(
        "--processed-parquet",
        default="data/processed/arena_full_features.parquet",
        help="Processed full feature table output path.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    raw, source = load_arena_raw(repo_root=repo_root, limit=args.limit, prefer_local=True)
    flat = flatten_initial_features(raw)

    processed_path = (repo_root / args.processed_parquet).resolve()
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    flat.to_parquet(processed_path, index=False)

    bool_columns = [
        "is_code",
        "creative_writing",
        "instruction_following",
        "math",
        "criteria_complexity",
        "criteria_creativity",
        "criteria_domain_knowledge",
        "criteria_problem_solving",
        "criteria_real_world",
        "criteria_specificity",
        "criteria_technical_accuracy",
        "is_binary_vote",
    ]
    numeric_columns = [
        "evaluation_order",
        "assistant_a_tokens",
        "assistant_b_tokens",
        "context_a_tokens",
        "context_b_tokens",
        "turns",
        "winner_binary",
        "length_diff_tokens",
        "abs_length_diff_tokens",
        "log_length_ratio",
        "length_diff_z",
    ]

    feature_cardinality = _feature_cardinality(flat)
    numeric_summary = _numeric_summary(flat, numeric_columns)
    boolean_summary = _boolean_summary(flat, bool_columns)
    model_frequency = _combined_model_frequency(flat)
    session_summary, session_size_distribution = _session_summary(flat)

    winner_counts = _top_counts(flat["winner"], top_n=20, normalize=False)
    winner_counts["share"] = winner_counts["count"] / winner_counts["count"].sum()
    language_counts = _top_counts(flat["language"], top_n=20, normalize=False)
    language_counts["share"] = language_counts["count"] / language_counts["count"].sum()
    task_bucket_counts = _top_counts(flat["task_bucket"], top_n=10, normalize=False)
    task_bucket_counts["share"] = task_bucket_counts["count"] / task_bucket_counts["count"].sum()
    evaluation_order_counts = _top_counts(flat["evaluation_order"], top_n=20, normalize=False)
    evaluation_order_counts["share"] = (
        evaluation_order_counts["count"] / evaluation_order_counts["count"].sum()
    )

    assistant_a_length_buckets = _length_bucket_counts(flat, "assistant_a_tokens")
    assistant_b_length_buckets = _length_bucket_counts(flat, "assistant_b_tokens")
    winner_by_task = _winner_by_group(flat, "task_bucket")
    winner_by_code = _winner_by_group(flat, "is_code")
    length_diff_by_winner, binary_winner_given_longer = _length_outcome_tables(flat)

    results_dir = (repo_root / args.output_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    feature_cardinality.to_csv(results_dir / "eda_feature_cardinality.csv", index=False)
    numeric_summary.to_csv(results_dir / "eda_numeric_summary.csv", index=False)
    boolean_summary.to_csv(results_dir / "eda_boolean_summary.csv", index=False)
    model_frequency.to_csv(results_dir / "eda_model_frequency.csv", index=False)
    session_summary.to_csv(results_dir / "eda_session_summary.csv", index=False)
    session_size_distribution.to_csv(results_dir / "eda_session_size_distribution.csv", index=False)
    winner_counts.to_csv(results_dir / "eda_winner_counts.csv", index=False)
    language_counts.to_csv(results_dir / "eda_language_counts.csv", index=False)
    task_bucket_counts.to_csv(results_dir / "eda_task_bucket_counts.csv", index=False)
    evaluation_order_counts.to_csv(results_dir / "eda_evaluation_order_counts.csv", index=False)
    assistant_a_length_buckets.to_csv(results_dir / "eda_assistant_a_length_buckets.csv", index=False)
    assistant_b_length_buckets.to_csv(results_dir / "eda_assistant_b_length_buckets.csv", index=False)
    winner_by_task.to_csv(results_dir / "eda_winner_by_task_bucket.csv", index=False)
    winner_by_code.to_csv(results_dir / "eda_winner_by_is_code.csv", index=False)
    length_diff_by_winner.to_csv(results_dir / "eda_length_diff_by_winner.csv", index=False)
    binary_winner_given_longer.to_csv(
        results_dir / "eda_binary_winner_given_longer.csv", index=False
    )

    report = build_markdown_report(
        source=source,
        flat=flat,
        feature_cardinality=feature_cardinality,
        numeric_summary=numeric_summary,
        boolean_summary=boolean_summary,
        model_frequency=model_frequency,
        session_summary=session_summary,
        session_size_distribution=session_size_distribution,
        winner_counts=winner_counts,
        language_counts=language_counts,
        task_bucket_counts=task_bucket_counts,
        evaluation_order_counts=evaluation_order_counts,
        assistant_a_length_buckets=assistant_a_length_buckets,
        assistant_b_length_buckets=assistant_b_length_buckets,
        winner_by_task=winner_by_task,
        winner_by_code=winner_by_code,
        length_diff_by_winner=length_diff_by_winner,
        binary_winner_given_longer=binary_winner_given_longer,
    )

    if args.output_markdown is None:
        report_path = results_dir / "full_dataset_eda.md"
    else:
        report_path = (repo_root / args.output_markdown).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"Wrote processed features to: {processed_path}")
    print(f"Wrote EDA report to: {report_path}")
    print(f"Saved supporting CSV tables to: {results_dir}")


if __name__ == "__main__":
    main()
