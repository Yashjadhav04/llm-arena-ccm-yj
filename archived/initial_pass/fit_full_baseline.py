from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.prepare_evaluation_splits import load_feature_table
from src.models.pairwise_preference import fit_pairwise_logit
from src.utils.arena_dataset import binary_vote_frame


def build_markdown_report(
    source: str,
    flat: pd.DataFrame,
    binary: pd.DataFrame,
    baseline,
) -> str:
    winner_counts = flat["winner"].fillna("unknown").astype(str).value_counts()
    language_counts = flat["language"].fillna("unknown").astype(str).value_counts().head(10)
    task_counts = flat["task_bucket"].fillna("unknown").astype(str).value_counts()

    lines: list[str] = []
    lines.append("# Full Dataset Baseline Model")
    lines.append("")
    lines.append("## Run Summary")
    lines.append(f"- Data source: {source}")
    lines.append(f"- Total rows available: {len(flat):,}")
    lines.append(f"- Binary rows used by the baseline model: {len(binary):,}")
    lines.append(f"- Unique models: {int(baseline.metrics['n_models'])}")
    lines.append("")
    lines.append("## Outcome Counts")
    for label, count in winner_counts.items():
        lines.append(f"- `{label}`: {int(count):,}")
    lines.append("")
    lines.append("## Task Bucket Counts")
    for label, count in task_counts.items():
        lines.append(f"- `{label}`: {int(count):,}")
    lines.append("")
    lines.append("## Top Languages")
    for label, count in language_counts.items():
        lines.append(f"- `{label}`: {int(count):,}")
    lines.append("")
    lines.append("## Baseline Metrics")
    lines.append(f"- Log loss: {baseline.metrics['log_loss']:.4f}")
    lines.append(f"- Accuracy: {baseline.metrics['accuracy']:.4f}")
    lines.append(f"- Null log loss: {baseline.metrics['null_log_loss']:.4f}")
    lines.append(f"- Mean model A win rate: {baseline.metrics['mean_model_a_win_rate']:.4f}")
    lines.append(f"- Reference model: `{baseline.metrics['reference_model']}`")
    lines.append("")
    lines.append("## Top Models By Baseline Score")
    for row in baseline.model_scores.head(15).itertuples(index=False):
        lines.append(f"- `{row.model}`: {row.score:.4f}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- This run uses only binary preference outcomes (`model_a` vs `model_b`).")
    lines.append("- `tie` and `both_bad` rows remain available for a later multinomial extension.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a descriptive baseline pairwise model on all available binary Arena data."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root containing data/, src/, and results/ directories.",
    )
    parser.add_argument(
        "--processed-parquet",
        default="data/processed/arena_full_features.parquet",
        help="Processed feature table to reuse or write.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/final_full_data_baseline",
        help="Directory for the descriptive full-data baseline outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row cap for smoke-testing. Default uses the full dataset.",
    )
    parser.add_argument(
        "--dataset-id",
        default="lmarena-ai/arena-human-preference-140k",
        help="Hugging Face dataset id to use when no local parquet files exist.",
    )
    parser.add_argument(
        "--prefer-local",
        action="store_true",
        help="Prefer local parquet files under data/raw/ when available.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=1000,
        help="Maximum L-BFGS iterations for the pairwise fit.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    processed_parquet = (repo_root / args.processed_parquet).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    flat, source = load_feature_table(
        repo_root=repo_root,
        processed_parquet=processed_parquet,
        limit=args.limit,
        dataset_id=args.dataset_id,
        prefer_local=args.prefer_local,
    )
    binary = binary_vote_frame(flat)
    baseline = fit_pairwise_logit(binary, maxiter=args.maxiter)

    metrics = pd.DataFrame([baseline.metrics])
    baseline.model_scores.to_csv(output_dir / "baseline_leaderboard.csv", index=False)
    metrics.to_csv(output_dir / "baseline_metrics.csv", index=False)

    report = build_markdown_report(source=source, flat=flat, binary=binary, baseline=baseline)
    report_path = output_dir / "baseline_summary.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"Wrote baseline leaderboard to: {output_dir / 'baseline_leaderboard.csv'}")
    print(f"Wrote baseline metrics to: {output_dir / 'baseline_metrics.csv'}")
    print(f"Wrote baseline summary to: {report_path}")


if __name__ == "__main__":
    main()
