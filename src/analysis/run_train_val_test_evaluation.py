from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.dataset_splits import (
    FixedBinarySplits,
    assign_session_splits,
    load_feature_table,
    load_fixed_binary_splits,
    materialize_split_datasets,
)
from src.models.pairwise_preference import (
    evaluate_pairwise_logit,
    fit_pairwise_logit,
)
from src.utils.arena_dataset import binary_vote_frame


MODEL_SPECS = [
    {"model_name": "baseline", "feature_columns": []},
    {
        "model_name": "proxy_augmented",
        "feature_columns": ["side_a_indicator", "relative_response_length_z"],
    },
]


def prepare_proxy_features(split_frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    prepared = split_frame.copy()
    prepared["side_a_indicator"] = 1.0

    train_mask = prepared["split"] == "train"
    train_length_mean = float(prepared.loc[train_mask, "length_diff_tokens"].mean(skipna=True))
    train_length_std = float(prepared.loc[train_mask, "length_diff_tokens"].std(skipna=True))

    if np.isnan(train_length_std) or train_length_std == 0:
        prepared["relative_response_length_z"] = 0.0
    else:
        prepared["relative_response_length_z"] = (
            prepared["length_diff_tokens"] - train_length_mean
        ) / train_length_std

    transform_stats = {
        "train_length_mean": train_length_mean,
        "train_length_std": train_length_std,
    }
    return prepared, transform_stats


def split_summary_table(split_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total_rows = len(split_frame)
    total_sessions = split_frame["evaluation_session_id"].nunique()

    for split_name in ["train", "validation", "test"]:
        subset = split_frame.loc[split_frame["split"] == split_name]
        rows.append(
            {
                "split": split_name,
                "rows": int(len(subset)),
                "row_share": float(len(subset) / total_rows),
                "sessions": int(subset["evaluation_session_id"].nunique()),
                "session_share": float(subset["evaluation_session_id"].nunique() / total_sessions),
                "model_a_win_rate": float(subset["winner_binary"].mean()),
            }
        )

    return pd.DataFrame(rows)


def fit_and_evaluate_models(
    split_frame: pd.DataFrame,
    maxiter: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    metrics_rows: list[dict[str, object]] = []
    fit_artifacts: dict[str, object] = {}

    for spec in MODEL_SPECS:
        model_name = spec["model_name"]
        feature_columns = spec["feature_columns"]
        train = split_frame.loc[split_frame["split"] == "train"].copy()
        fitted = fit_pairwise_logit(train, feature_columns=feature_columns, maxiter=maxiter)
        fit_artifacts[model_name] = fitted

        for split_name in ["train", "validation", "test"]:
            subset = split_frame.loc[split_frame["split"] == split_name].copy()
            metrics = evaluate_pairwise_logit(subset, fitted, unknown_policy="drop")
            metrics_rows.append(
                {
                    "model_name": model_name,
                    "split": split_name,
                    **metrics,
                }
            )

    return pd.DataFrame(metrics_rows), fit_artifacts


def build_markdown_report(
    source: str,
    split_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    proxy_coefficients: pd.DataFrame,
    transform_stats: dict[str, float],
) -> str:
    lines: list[str] = []
    lines.append("# Train / Validation / Test Evaluation")
    lines.append("")
    lines.append("## Data Source")
    lines.append(f"- {source}")
    lines.append("")
    lines.append("## Split Summary")
    for row in split_summary.itertuples(index=False):
        lines.append(
            f"- `{row.split}`: rows={int(row.rows):,}, row_share={row.row_share:.4f}, "
            f"sessions={int(row.sessions):,}, model_a_win_rate={row.model_a_win_rate:.4f}"
        )
    lines.append("")
    lines.append("## Proxy Feature Scaling")
    lines.append(f"- Train-set length mean: {transform_stats['train_length_mean']:.4f}")
    lines.append(f"- Train-set length std: {transform_stats['train_length_std']:.4f}")
    lines.append("")
    lines.append("## Metrics By Model And Split")
    for split_name in ["train", "validation", "test"]:
        lines.append(f"### {split_name.title()}")
        subset = metrics.loc[metrics["split"] == split_name].copy()
        for row in subset.itertuples(index=False):
            lines.append(
                f"- `{row.model_name}`: log_loss={row.log_loss:.4f}, accuracy={row.accuracy:.4f}, "
                f"brier_score={row.brier_score:.4f}, covered_row_share={row.covered_row_share:.4f}"
            )
        lines.append("")

    validation = metrics.pivot(index="split", columns="model_name", values="log_loss")
    accuracy = metrics.pivot(index="split", columns="model_name", values="accuracy")
    brier = metrics.pivot(index="split", columns="model_name", values="brier_score")

    for split_name in ["validation", "test"]:
        lines.append(f"## {split_name.title()} Delta: Proxy Minus Baseline")
        log_loss_delta = validation.loc[split_name, "proxy_augmented"] - validation.loc[
            split_name, "baseline"
        ]
        accuracy_delta = accuracy.loc[split_name, "proxy_augmented"] - accuracy.loc[
            split_name, "baseline"
        ]
        brier_delta = brier.loc[split_name, "proxy_augmented"] - brier.loc[split_name, "baseline"]
        lines.append(f"- Log loss delta: {log_loss_delta:.4f}")
        lines.append(f"- Accuracy delta: {accuracy_delta:.4f}")
        lines.append(f"- Brier score delta: {brier_delta:.4f}")
        lines.append("")

    lines.append("## Proxy Coefficients")
    for row in proxy_coefficients.itertuples(index=False):
        lines.append(f"- `{row.term}`: {row.estimate:.4f}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Splits are session-level to reduce leakage across related votes.")
    lines.append("- Report test-set metrics as the main comparison in the write-up.")
    lines.append("- Use validation metrics for model/feature iteration before touching the test set.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a train/validation/test evaluation for the baseline and proxy-augmented models."
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
        default="results/train_val_test_evaluation",
        help="Directory for evaluation outputs.",
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
        "--train-share",
        type=float,
        default=0.8,
        help="Target share of rows to place in training via session-level assignment.",
    )
    parser.add_argument(
        "--validation-share",
        type=float,
        default=0.1,
        help="Target share of rows to place in validation via session-level assignment.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for the session-level split.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=1000,
        help="Maximum L-BFGS iterations for each pairwise fit.",
    )
    parser.add_argument(
        "--reuse-splits-from",
        default=None,
        help="Optional path to an existing split_assignments.csv file to reuse exactly.",
    )
    parser.add_argument(
        "--materialize-split-parquets",
        action="store_true",
        help="Also save train.parquet, validation.parquet, and test.parquet for convenient reuse.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    processed_parquet = (repo_root / args.processed_parquet).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_splits_from:
        fixed_splits: FixedBinarySplits = load_fixed_binary_splits(
            repo_root=repo_root,
            processed_parquet=processed_parquet,
            split_assignments_path=args.reuse_splits_from,
            limit=args.limit,
            dataset_id=args.dataset_id,
            prefer_local=args.prefer_local,
        )
        source = fixed_splits.source
        split_frame = fixed_splits.frame
        split_source = f"reused from {fixed_splits.split_source}"
        binary = split_frame.drop(columns=["split"]).copy()
    else:
        flat, source = load_feature_table(
            repo_root=repo_root,
            processed_parquet=processed_parquet,
            limit=args.limit,
            dataset_id=args.dataset_id,
            prefer_local=args.prefer_local,
        )
        binary = binary_vote_frame(flat)
        split_frame = assign_session_splits(
            binary=binary,
            train_share=args.train_share,
            validation_share=args.validation_share,
            seed=args.seed,
        )
        split_source = "generated in this run"

    split_frame, transform_stats = prepare_proxy_features(split_frame)

    split_summary = split_summary_table(split_frame)
    metrics, fit_artifacts = fit_and_evaluate_models(split_frame=split_frame, maxiter=args.maxiter)

    split_assignments = split_frame[["id", "evaluation_session_id", "split"]].copy()
    split_assignments.to_csv(output_dir / "split_assignments.csv", index=False)
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    pd.DataFrame([transform_stats]).to_csv(output_dir / "proxy_transform_stats.csv", index=False)
    metrics.to_csv(output_dir / "evaluation_metrics.csv", index=False)

    split_metadata = {
        "split_source": split_source,
        "processed_parquet": str(processed_parquet),
        "limit": args.limit,
        "train_share": args.train_share,
        "validation_share": args.validation_share,
        "seed": args.seed,
        "n_binary_rows": int(len(binary)),
    }
    (output_dir / "split_metadata.json").write_text(
        json.dumps(split_metadata, indent=2),
        encoding="utf-8",
    )

    if args.materialize_split_parquets:
        materialize_split_datasets(split_frame=split_frame, output_dir=output_dir)

    for model_name, fitted in fit_artifacts.items():
        fitted.model_scores.to_csv(output_dir / f"{model_name}_leaderboard.csv", index=False)
        coefficient_path = output_dir / f"{model_name}_coefficients.csv"
        if fitted.coefficients.empty:
            if coefficient_path.exists():
                coefficient_path.unlink()
        else:
            fitted.coefficients.to_csv(coefficient_path, index=False)

    report = build_markdown_report(
        source=source,
        split_summary=split_summary,
        metrics=metrics,
        proxy_coefficients=fit_artifacts["proxy_augmented"].coefficients,
        transform_stats=transform_stats,
    )
    report_path = output_dir / "evaluation_summary.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"Wrote split assignments to: {output_dir / 'split_assignments.csv'}")
    print(f"Wrote split summary to: {output_dir / 'split_summary.csv'}")
    print(f"Wrote evaluation metrics to: {output_dir / 'evaluation_metrics.csv'}")
    print(f"Wrote evaluation summary to: {report_path}")
    if args.materialize_split_parquets:
        print(f"Wrote split parquet files to: {output_dir}")


if __name__ == "__main__":
    main()
