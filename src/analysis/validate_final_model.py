from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.evaluate_preference_models import (
    MODEL_SPECS,
    add_signal_interaction_features,
    ensure_formatting_feature,
    ensure_proxy_features,
    load_split_parquets,
)
from src.models.pairwise_preference import (
    PairwisePreferenceResult,
    evaluate_pairwise_logit,
    fit_pairwise_logit,
    predict_pairwise_logit,
)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def prepare_splits(
    *,
    repo_root: Path,
    split_dir: Path,
    dataset_id: str,
    formatting_cache: Path,
    max_rows_per_split: int | None,
    subsample_seed: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    splits = load_split_parquets(split_dir)
    splits, transform_stats = ensure_proxy_features(splits)
    splits = ensure_formatting_feature(
        splits=splits,
        repo_root=repo_root,
        dataset_id=dataset_id,
        cache_path=formatting_cache,
    )
    splits = add_signal_interaction_features(splits)

    if max_rows_per_split is not None:
        sampled: dict[str, pd.DataFrame] = {}
        for split_name, frame in splits.items():
            if len(frame) <= max_rows_per_split:
                sampled[split_name] = frame.copy()
            else:
                sampled[split_name] = frame.sample(
                    n=max_rows_per_split,
                    random_state=subsample_seed,
                ).copy()
        splits = sampled

    return splits, transform_stats


def generate_synthetic_split(
    frame: pd.DataFrame,
    generator_result: PairwisePreferenceResult,
    rng: np.random.Generator,
) -> pd.DataFrame:
    predictions = predict_pairwise_logit(frame, generator_result)
    probs = predictions["pred_proba"].to_numpy(dtype=float)
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    sampled = rng.binomial(1, probs).astype(float)

    out = frame.copy()
    out["winner_binary"] = sampled
    return out


def generate_synthetic_splits(
    splits: dict[str, pd.DataFrame],
    generator_result: PairwisePreferenceResult,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    return {
        split_name: generate_synthetic_split(frame, generator_result, rng)
        for split_name, frame in splits.items()
    }


def summarize_parameter_recovery(
    *,
    generator_model: str,
    rep: int,
    true_result: PairwisePreferenceResult,
    recovered_result: PairwisePreferenceResult,
) -> dict[str, float | int | str]:
    true_scores = true_result.model_scores.set_index("model")["score"]
    est_scores = recovered_result.model_scores.set_index("model")["score"]
    common_models = sorted(set(true_scores.index).intersection(set(est_scores.index)))

    score_true = true_scores.loc[common_models].to_numpy(dtype=float)
    score_est = est_scores.loc[common_models].to_numpy(dtype=float)

    # Skills are identifiable up to a constant shift, so compare centered vectors.
    score_true_centered = score_true - score_true.mean()
    score_est_centered = score_est - score_est.mean()

    skill_corr = _safe_corr(score_true_centered, score_est_centered)
    skill_rmse = float(
        np.sqrt(np.mean((score_est_centered - score_true_centered) ** 2))
    )
    skill_mae = float(np.mean(np.abs(score_est_centered - score_true_centered)))

    true_beta = true_result.coefficients.set_index("term")["estimate"]
    est_beta = recovered_result.coefficients.set_index("term")["estimate"]
    common_terms = sorted(set(true_beta.index).intersection(set(est_beta.index)))

    if common_terms:
        beta_true = true_beta.loc[common_terms].to_numpy(dtype=float)
        beta_est = est_beta.loc[common_terms].to_numpy(dtype=float)
        beta_corr = _safe_corr(beta_true, beta_est)
        beta_rmse = float(np.sqrt(np.mean((beta_est - beta_true) ** 2)))
        beta_mae = float(np.mean(np.abs(beta_est - beta_true)))
    else:
        beta_corr = float("nan")
        beta_rmse = float("nan")
        beta_mae = float("nan")

    return {
        "generator_model": generator_model,
        "rep": rep,
        "n_models_compared": len(common_models),
        "n_beta_terms_compared": len(common_terms),
        "skill_corr": skill_corr,
        "skill_rmse": skill_rmse,
        "skill_mae": skill_mae,
        "beta_corr": beta_corr,
        "beta_rmse": beta_rmse,
        "beta_mae": beta_mae,
    }


def _spec_map() -> dict[str, list[str]]:
    return {
        spec["model_name"]: list(spec["feature_columns"])  # defensive copy
        for spec in MODEL_SPECS
    }


def _candidate_rows(
    *,
    synthetic_splits: dict[str, pd.DataFrame],
    candidate_specs: list[tuple[str, list[str]]],
    maxiter: int,
    selection_split: str,
    selection_metric: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    train = synthetic_splits["train"]
    heldout = synthetic_splits[selection_split]
    for model_name, feature_columns in candidate_specs:
        fitted = fit_pairwise_logit(
            train,
            feature_columns=feature_columns,
            maxiter=maxiter,
        )
        metrics = evaluate_pairwise_logit(heldout, fitted, unknown_policy="drop")
        rows.append({"fit_model": model_name, **metrics})

    out = pd.DataFrame(rows)
    if selection_metric not in out.columns:
        raise ValueError(f"Selection metric '{selection_metric}' not in candidate metrics.")
    best_idx = out[selection_metric].astype(float).idxmin()
    out["is_selected"] = False
    out.loc[best_idx, "is_selected"] = True
    out["selected_model"] = str(out.loc[best_idx, "fit_model"])
    return out


def _build_summary_markdown(
    *,
    final_model: str,
    n_reps: int,
    selection_split: str,
    selection_metric: str,
    transform_stats: dict[str, float],
    parameter_summary: pd.DataFrame,
    selection_rates: pd.DataFrame,
    candidate_means: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# Final Model Validation Summary")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- Final generator model: `{final_model}`")
    lines.append(f"- Replications: {n_reps}")
    lines.append(f"- Model selection split: `{selection_split}`")
    lines.append(f"- Model selection metric: `{selection_metric}` (lower is better)")
    lines.append(
        f"- Train length scaling: mean={transform_stats['train_length_mean']:.4f}, "
        f"std={transform_stats['train_length_std']:.4f}"
    )
    lines.append("")

    lines.append("## Parameter Recovery")
    row = parameter_summary.iloc[0]
    lines.append(
        f"- Skill correlation: {row['skill_corr_mean']:.4f} ± {row['skill_corr_std']:.4f}"
    )
    lines.append(
        f"- Skill RMSE: {row['skill_rmse_mean']:.4f} ± {row['skill_rmse_std']:.4f}"
    )
    lines.append(
        f"- Coefficient correlation: {row['beta_corr_mean']:.4f} ± {row['beta_corr_std']:.4f}"
    )
    lines.append(
        f"- Coefficient RMSE: {row['beta_rmse_mean']:.4f} ± {row['beta_rmse_std']:.4f}"
    )
    lines.append("")

    lines.append("## Model Recovery (Selection Rates)")
    for row in selection_rates.itertuples(index=False):
        lines.append(
            f"- `{row.fit_model}` selected in {int(row.count)}/{n_reps} reps "
            f"({row.selection_rate:.3f})"
        )
    lines.append("")

    lines.append("## Candidate Held-out Means")
    for row in candidate_means.itertuples(index=False):
        lines.append(
            f"- `{row.fit_model}`: log_loss={row.log_loss:.4f}, "
            f"accuracy={row.accuracy:.4f}, brier_score={row.brier_score:.4f}"
        )
    lines.append("")

    lines.append("## Notes")
    lines.append(
        "- This run validates only the final generator model, not a full confusion matrix across all generators."
    )
    lines.append(
        "- Strong validation evidence is: high parameter recovery and high self-selection rate for the final model."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run focused model validation for one final model: "
            "parameter recovery + candidate model recovery selection."
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
        default="results/final_model_validation_30reps",
        help="Directory for outputs.",
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
            "Optional formatting feature cache. "
            "If omitted, defaults to output-dir/formatting_features_cache.parquet."
        ),
    )
    parser.add_argument(
        "--final-model",
        default="full_formatting",
        help="Model name to use as synthetic data generator and recovery target.",
    )
    parser.add_argument(
        "--candidate-models",
        default="baseline,full_formatting,full_formatting_plus_signals,signal_interactions",
        help="Comma-separated model names to include in candidate model recovery fits.",
    )
    parser.add_argument(
        "--n-reps",
        type=int,
        default=30,
        help="Number of simulation replications.",
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
        default=29,
        help="Random seed for Bernoulli sampling in simulations.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=1500,
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

    spec_map = _spec_map()
    if args.final_model not in spec_map:
        raise ValueError(
            f"Unknown final model '{args.final_model}'. "
            f"Available: {sorted(spec_map.keys())}"
        )

    candidate_models = [name.strip() for name in args.candidate_models.split(",") if name.strip()]
    unknown_candidates = [name for name in candidate_models if name not in spec_map]
    if unknown_candidates:
        raise ValueError(
            f"Unknown candidate model(s): {unknown_candidates}. "
            f"Available: {sorted(spec_map.keys())}"
        )
    candidate_specs = [(name, spec_map[name]) for name in candidate_models]

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

    print(f"Fitting final generator model: {args.final_model}")
    true_result = fit_pairwise_logit(
        splits["train"],
        feature_columns=spec_map[args.final_model],
        maxiter=args.maxiter,
    )
    true_result.model_scores.to_csv(
        output_dir / f"generator_{args.final_model}_leaderboard.csv",
        index=False,
    )
    if not true_result.coefficients.empty:
        true_result.coefficients.to_csv(
            output_dir / f"generator_{args.final_model}_coefficients.csv",
            index=False,
        )

    print("Running simulation replications...")
    rng = np.random.default_rng(args.seed)
    parameter_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    for rep in range(args.n_reps):
        synthetic_splits = generate_synthetic_splits(splits, true_result, rng)

        recovered = fit_pairwise_logit(
            synthetic_splits["train"],
            feature_columns=spec_map[args.final_model],
            maxiter=args.maxiter,
        )
        parameter_rows.append(
            summarize_parameter_recovery(
                generator_model=args.final_model,
                rep=rep,
                true_result=true_result,
                recovered_result=recovered,
            )
        )

        candidate_scores = _candidate_rows(
            synthetic_splits=synthetic_splits,
            candidate_specs=candidate_specs,
            maxiter=args.maxiter,
            selection_split=args.selection_split,
            selection_metric=args.selection_metric,
        )
        candidate_scores["rep"] = rep
        model_rows.extend(candidate_scores.to_dict(orient="records"))

    parameter_runs = pd.DataFrame(parameter_rows)
    model_scores = pd.DataFrame(model_rows)

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

    selected = model_scores.loc[model_scores["is_selected"]].copy()
    selection_rates = (
        selected["fit_model"]
        .value_counts()
        .rename_axis("fit_model")
        .reset_index(name="count")
        .sort_values(["count", "fit_model"], ascending=[False, True])
        .reset_index(drop=True)
    )
    selection_rates["selection_rate"] = selection_rates["count"] / float(args.n_reps)

    candidate_means = (
        model_scores.groupby("fit_model", as_index=False)[["log_loss", "accuracy", "brier_score"]]
        .mean()
        .sort_values("log_loss")
        .reset_index(drop=True)
    )

    parameter_runs.to_csv(output_dir / "parameter_recovery_runs.csv", index=False)
    parameter_summary.to_csv(output_dir / "parameter_recovery_summary.csv", index=False)
    model_scores.to_csv(output_dir / "model_recovery_scores.csv", index=False)
    selected.to_csv(output_dir / "model_recovery_selected.csv", index=False)
    selection_rates.to_csv(output_dir / "model_recovery_selection_rates.csv", index=False)
    candidate_means.to_csv(output_dir / "model_recovery_candidate_means.csv", index=False)

    summary = _build_summary_markdown(
        final_model=args.final_model,
        n_reps=args.n_reps,
        selection_split=args.selection_split,
        selection_metric=args.selection_metric,
        transform_stats=transform_stats,
        parameter_summary=parameter_summary,
        selection_rates=selection_rates,
        candidate_means=candidate_means,
    )
    summary_path = output_dir / "validation_summary.md"
    summary_path.write_text(summary, encoding="utf-8")

    print(f"Wrote: {output_dir / 'parameter_recovery_runs.csv'}")
    print(f"Wrote: {output_dir / 'parameter_recovery_summary.csv'}")
    print(f"Wrote: {output_dir / 'model_recovery_scores.csv'}")
    print(f"Wrote: {output_dir / 'model_recovery_selected.csv'}")
    print(f"Wrote: {output_dir / 'model_recovery_selection_rates.csv'}")
    print(f"Wrote: {output_dir / 'model_recovery_candidate_means.csv'}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
