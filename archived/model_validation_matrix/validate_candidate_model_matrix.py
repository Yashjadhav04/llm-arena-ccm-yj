from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.evaluate_preference_models import MODEL_SPECS
from src.analysis.validate_final_model import (
    generate_synthetic_splits,
    prepare_splits,
    summarize_parameter_recovery,
)
from src.models.pairwise_preference import (
    PairwisePreferenceResult,
    evaluate_pairwise_logit,
    fit_pairwise_logit,
)


def fit_reference_generators(train: pd.DataFrame, maxiter: int) -> dict[str, PairwisePreferenceResult]:
    generators: dict[str, PairwisePreferenceResult] = {}
    for spec in MODEL_SPECS:
        model_name = spec["model_name"]
        feature_columns = spec["feature_columns"]
        generators[model_name] = fit_pairwise_logit(
            train,
            feature_columns=feature_columns,
            maxiter=maxiter,
        )
    return generators


def run_model_recovery_for_rep(
    *,
    generator_model: str,
    rep: int,
    synthetic_splits: dict[str, pd.DataFrame],
    maxiter: int,
    selection_split: str,
    selection_metric: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | bool]] = []
    train = synthetic_splits["train"]
    heldout = synthetic_splits[selection_split]

    for spec in MODEL_SPECS:
        fit_model = spec["model_name"]
        feature_columns = spec["feature_columns"]
        fitted = fit_pairwise_logit(train, feature_columns=feature_columns, maxiter=maxiter)
        metrics = evaluate_pairwise_logit(heldout, fitted, unknown_policy="drop")
        rows.append(
            {
                "generator_model": generator_model,
                "rep": rep,
                "fit_model": fit_model,
                **metrics,
            }
        )

    out = pd.DataFrame(rows)
    if selection_metric not in out.columns:
        raise ValueError(
            f"Selection metric '{selection_metric}' not found. "
            f"Available: {sorted(out.columns)}"
        )
    best_idx = out[selection_metric].astype(float).idxmin()
    out["is_selected"] = False
    out.loc[best_idx, "is_selected"] = True
    selected = str(out.loc[best_idx, "fit_model"])
    out["selected_model"] = selected
    return out


def summarize_recovery(
    *,
    parameter_runs: pd.DataFrame,
    model_runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parameter_summary = (
        parameter_runs.groupby("generator_model", as_index=False)
        .agg(
            n_reps=("rep", "nunique"),
            skill_corr_mean=("skill_corr", "mean"),
            skill_corr_std=("skill_corr", "std"),
            skill_rmse_mean=("skill_rmse", "mean"),
            skill_rmse_std=("skill_rmse", "std"),
            beta_corr_mean=("beta_corr", "mean"),
            beta_corr_std=("beta_corr", "std"),
            beta_rmse_mean=("beta_rmse", "mean"),
            beta_rmse_std=("beta_rmse", "std"),
        )
        .sort_values("generator_model")
        .reset_index(drop=True)
    )

    model_names = [spec["model_name"] for spec in MODEL_SPECS]
    selected = model_runs.loc[model_runs["is_selected"]].copy()

    count_matrix = pd.crosstab(
        selected["generator_model"],
        selected["fit_model"],
    ).reindex(index=model_names, columns=model_names, fill_value=0)
    count_matrix.index.name = "generator_model"
    confusion_counts = count_matrix.reset_index()

    row_totals = count_matrix.sum(axis=1).replace(0, np.nan)
    prob_matrix = count_matrix.div(row_totals, axis=0).fillna(0.0)
    prob_matrix.index.name = "generator_model"
    confusion_probs = prob_matrix.reset_index()

    return parameter_summary, confusion_counts, confusion_probs


def build_summary_markdown(
    *,
    parameter_summary: pd.DataFrame,
    confusion_probs: pd.DataFrame,
    model_runs: pd.DataFrame,
    transform_stats: dict[str, float],
    n_reps: int,
    selection_split: str,
    selection_metric: str,
) -> str:
    lines: list[str] = []
    lines.append("# Model Validation Summary")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- Replications per generator model: {n_reps}")
    lines.append(f"- Model selection split: `{selection_split}`")
    lines.append(f"- Model selection metric: `{selection_metric}` (lower is better)")
    lines.append(
        f"- Train length scaling: mean={transform_stats['train_length_mean']:.4f}, "
        f"std={transform_stats['train_length_std']:.4f}"
    )
    lines.append("")

    lines.append("## Parameter Recovery (Across Replications)")
    for row in parameter_summary.itertuples(index=False):
        lines.append(
            f"- `{row.generator_model}`: skill_corr={row.skill_corr_mean:.3f}±{row.skill_corr_std:.3f}, "
            f"skill_rmse={row.skill_rmse_mean:.3f}±{row.skill_rmse_std:.3f}, "
            f"beta_corr={row.beta_corr_mean:.3f}±{row.beta_corr_std:.3f}, "
            f"beta_rmse={row.beta_rmse_mean:.3f}±{row.beta_rmse_std:.3f}"
        )
    lines.append("")

    lines.append("## Model Recovery Confusion Matrix (Row-Normalized)")
    prob_indexed = confusion_probs.set_index("generator_model")
    model_names = [spec["model_name"] for spec in MODEL_SPECS]
    diagonal_values: list[float] = []
    for generator_model in model_names:
        if generator_model not in prob_indexed.index:
            continue
        parts = []
        for fit_model in model_names:
            value = float(prob_indexed.loc[generator_model].get(fit_model, 0.0))
            parts.append(f"{fit_model}={value:.3f}")
            if fit_model == generator_model:
                diagonal_values.append(value)
        lines.append(f"- `{generator_model}` -> " + ", ".join(parts))
    if diagonal_values:
        lines.append(
            f"- Mean diagonal probability: {float(np.mean(diagonal_values)):.3f}"
        )
    lines.append("")

    lines.append("## Held-out Fit Averages (By Generator / Candidate)")
    grouped = (
        model_runs.groupby(["generator_model", "fit_model"], as_index=False)[
            ["log_loss", "accuracy", "brier_score"]
        ]
        .mean()
        .sort_values(["generator_model", "log_loss", "fit_model"])
    )
    for generator_model in model_names:
        subset = grouped.loc[grouped["generator_model"] == generator_model]
        if subset.empty:
            continue
        line = ", ".join(
            [
                f"{row.fit_model}: log_loss={row.log_loss:.4f}, acc={row.accuracy:.4f}"
                for row in subset.itertuples(index=False)
            ]
        )
        lines.append(f"- `{generator_model}`: {line}")
    lines.append("")
    lines.append("## Notes")
    lines.append(
        "- Parameter recovery compares centered skill vectors (shift-invariant identification)."
    )
    lines.append(
        "- For model recovery, each synthetic dataset is fit by all candidate models; "
        "the selected model minimizes held-out log loss."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run simulation-based model validation: parameter recovery and "
            "model-recovery confusion matrix."
        )
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--split-dir",
        default="results/train_val_test_evaluation",
        help="Directory containing train/validation/test parquet splits.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/model_validation",
        help="Directory for recovery outputs.",
    )
    parser.add_argument(
        "--dataset-id",
        default="lmarena-ai/arena-human-preference-140k",
        help="HF dataset id (used only if formatting cache cannot be built from local raw data).",
    )
    parser.add_argument(
        "--formatting-cache",
        default=None,
        help=(
            "Optional parquet cache with id plus formatting feature columns. "
            "If omitted, defaults to output-dir/formatting_features_cache.parquet."
        ),
    )
    parser.add_argument(
        "--n-reps",
        type=int,
        default=10,
        help="Number of simulation replications per generator model.",
    )
    parser.add_argument(
        "--selection-split",
        default="validation",
        choices=["validation", "test"],
        help="Held-out split used for model-recovery selection.",
    )
    parser.add_argument(
        "--selection-metric",
        default="log_loss",
        choices=["log_loss", "brier_score"],
        help="Metric used to select the winning model per synthetic dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for Bernoulli sampling in simulations.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=1000,
        help="Maximum L-BFGS iterations for each fit.",
    )
    parser.add_argument(
        "--max-rows-per-split",
        type=int,
        default=None,
        help="Optional cap for each split (useful for smoke runs).",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    split_dir = (repo_root / args.split_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.formatting_cache:
        formatting_cache = (repo_root / args.formatting_cache).resolve()
    else:
        formatting_cache = (output_dir / "formatting_features_cache.parquet").resolve()

    print("Preparing split data and features...")
    splits, transform_stats = prepare_splits(
        repo_root=repo_root,
        split_dir=split_dir,
        dataset_id=args.dataset_id,
        formatting_cache=formatting_cache,
        max_rows_per_split=args.max_rows_per_split,
        subsample_seed=args.seed,
    )
    for split_name, frame in splits.items():
        print(f"  {split_name}: {len(frame):,} rows")

    print("Fitting reference generator parameters from the train split...")
    reference = fit_reference_generators(splits["train"], maxiter=args.maxiter)

    for model_name, fitted in reference.items():
        fitted.model_scores.to_csv(
            output_dir / f"generator_{model_name}_leaderboard.csv",
            index=False,
        )
        if not fitted.coefficients.empty:
            fitted.coefficients.to_csv(
                output_dir / f"generator_{model_name}_coefficients.csv",
                index=False,
            )

    rng = np.random.default_rng(args.seed)
    parameter_rows: list[dict[str, float | int | str]] = []
    model_rows: list[dict[str, float | int | str | bool]] = []

    print("Running simulation replications...")
    for spec in MODEL_SPECS:
        generator_model = spec["model_name"]
        feature_columns = spec["feature_columns"]
        true_result = reference[generator_model]
        print(f"  Generator: {generator_model}")

        for rep in range(args.n_reps):
            synthetic_splits = generate_synthetic_splits(splits, true_result, rng)

            recovered = fit_pairwise_logit(
                synthetic_splits["train"],
                feature_columns=feature_columns,
                maxiter=args.maxiter,
            )
            parameter_rows.append(
                summarize_parameter_recovery(
                    generator_model=generator_model,
                    rep=rep,
                    true_result=true_result,
                    recovered_result=recovered,
                )
            )

            model_run = run_model_recovery_for_rep(
                generator_model=generator_model,
                rep=rep,
                synthetic_splits=synthetic_splits,
                maxiter=args.maxiter,
                selection_split=args.selection_split,
                selection_metric=args.selection_metric,
            )
            model_rows.extend(model_run.to_dict(orient="records"))

    parameter_runs = pd.DataFrame(parameter_rows)
    model_runs = pd.DataFrame(model_rows)

    parameter_summary, confusion_counts, confusion_probs = summarize_recovery(
        parameter_runs=parameter_runs,
        model_runs=model_runs,
    )

    selected = model_runs.loc[model_runs["is_selected"]].copy()
    selected = selected.rename(columns={"fit_model": "selected_model"})
    selected = selected[
        [
            "generator_model",
            "rep",
            "selected_model",
            "log_loss",
            "accuracy",
            "brier_score",
            "n_rows",
            "covered_row_share",
        ]
    ]

    parameter_runs.to_csv(output_dir / "parameter_recovery_runs.csv", index=False)
    parameter_summary.to_csv(output_dir / "parameter_recovery_summary.csv", index=False)
    model_runs.to_csv(output_dir / "model_recovery_scores.csv", index=False)
    selected.to_csv(output_dir / "model_recovery_selected.csv", index=False)
    confusion_counts.to_csv(output_dir / "model_recovery_confusion_counts.csv", index=False)
    confusion_probs.to_csv(output_dir / "model_recovery_confusion_row_normalized.csv", index=False)

    summary = build_summary_markdown(
        parameter_summary=parameter_summary,
        confusion_probs=confusion_probs,
        model_runs=model_runs,
        transform_stats=transform_stats,
        n_reps=args.n_reps,
        selection_split=args.selection_split,
        selection_metric=args.selection_metric,
    )
    summary_path = output_dir / "validation_summary.md"
    summary_path.write_text(summary, encoding="utf-8")

    print(f"Wrote: {output_dir / 'parameter_recovery_runs.csv'}")
    print(f"Wrote: {output_dir / 'parameter_recovery_summary.csv'}")
    print(f"Wrote: {output_dir / 'model_recovery_scores.csv'}")
    print(f"Wrote: {output_dir / 'model_recovery_selected.csv'}")
    print(f"Wrote: {output_dir / 'model_recovery_confusion_counts.csv'}")
    print(f"Wrote: {output_dir / 'model_recovery_confusion_row_normalized.csv'}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
