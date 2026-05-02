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
from src.models.pairwise_preference import fit_pairwise_logit, predict_pairwise_logit


def _spec_map() -> dict[str, list[str]]:
    return {
        spec["model_name"]: list(spec["feature_columns"])  # defensive copy
        for spec in MODEL_SPECS
    }


def _session_metric_table(
    frame: pd.DataFrame,
    model_result: object,
    session_col: str,
) -> pd.DataFrame:
    pred = predict_pairwise_logit(frame, model_result)
    y = pd.to_numeric(frame["winner_binary"], errors="coerce").to_numpy(dtype=float)
    known = pred["known_pair"].to_numpy(dtype=bool)
    mask = (~np.isnan(y)) & known

    probs = pred.loc[mask, "pred_proba"].to_numpy(dtype=float)
    labels = pred.loc[mask, "pred_label"].to_numpy(dtype=float)
    y_valid = y[mask]
    sessions = frame.loc[mask, session_col].astype(str).to_numpy()

    eps = 1e-9
    loss_item = -(y_valid * np.log(probs + eps) + (1.0 - y_valid) * np.log(1.0 - probs + eps))
    brier_item = (probs - y_valid) ** 2
    acc_item = (labels == y_valid).astype(float)

    tmp = pd.DataFrame(
        {
            "session_id": sessions,
            "n_rows": 1.0,
            "loss_sum": loss_item,
            "brier_sum": brier_item,
            "acc_sum": acc_item,
        }
    )
    grouped = (
        tmp.groupby("session_id", as_index=False)[["n_rows", "loss_sum", "brier_sum", "acc_sum"]]
        .sum()
        .sort_values("session_id")
        .reset_index(drop=True)
    )
    return grouped


def _align_session_arrays(
    grouped: pd.DataFrame,
    session_index: pd.Index,
) -> dict[str, np.ndarray]:
    aligned = grouped.set_index("session_id").reindex(session_index, fill_value=0.0)
    return {
        "n_rows": aligned["n_rows"].to_numpy(dtype=float),
        "loss_sum": aligned["loss_sum"].to_numpy(dtype=float),
        "brier_sum": aligned["brier_sum"].to_numpy(dtype=float),
        "acc_sum": aligned["acc_sum"].to_numpy(dtype=float),
    }


def _metrics_from_sample(arrays: dict[str, np.ndarray], sample_idx: np.ndarray) -> dict[str, float]:
    total_n = float(np.sum(arrays["n_rows"][sample_idx]))
    if total_n <= 0:
        return {"log_loss": float("nan"), "brier_score": float("nan"), "accuracy": float("nan")}
    return {
        "log_loss": float(np.sum(arrays["loss_sum"][sample_idx]) / total_n),
        "brier_score": float(np.sum(arrays["brier_sum"][sample_idx]) / total_n),
        "accuracy": float(np.sum(arrays["acc_sum"][sample_idx]) / total_n),
    }


def _summarize_samples(samples: pd.Series) -> dict[str, float]:
    valid = samples.dropna().to_numpy(dtype=float)
    if valid.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0,
        "ci_lower": float(np.quantile(valid, 0.025)),
        "ci_upper": float(np.quantile(valid, 0.975)),
    }


def build_summary_markdown(
    model_ci: pd.DataFrame,
    delta_ci: pd.DataFrame,
    n_bootstrap: int,
    compared_models: list[str],
) -> str:
    lines: list[str] = []
    lines.append("# Metric Uncertainty (Session Bootstrap)")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- Bootstrap replications: {n_bootstrap}")
    lines.append("- Resampling unit: evaluation session (with replacement)")
    lines.append(f"- Compared models: {', '.join([f'`{m}`' for m in compared_models])}")
    lines.append("")

    lines.append("## Model Metric CIs")
    for split_name in ["validation", "test"]:
        lines.append(f"### {split_name.title()}")
        subset = model_ci.loc[model_ci["split"] == split_name].copy()
        for metric_name in ["log_loss", "accuracy", "brier_score"]:
            lines.append(f"- `{metric_name}`")
            metric_rows = subset.loc[subset["metric"] == metric_name]
            for row in metric_rows.itertuples(index=False):
                lines.append(
                    f"  `{row.model_name}`: mean={row.mean:.6f}, "
                    f"95% CI [{row.ci_lower:.6f}, {row.ci_upper:.6f}]"
                )
        lines.append("")

    lines.append("## Delta CIs")
    lines.append("- Delta definition: `model_a - model_b`")
    for split_name in ["validation", "test"]:
        lines.append(f"### {split_name.title()}")
        subset = delta_ci.loc[delta_ci["split"] == split_name].copy()
        for metric_name in ["log_loss", "accuracy", "brier_score"]:
            metric_rows = subset.loc[subset["metric"] == metric_name]
            if metric_rows.empty:
                continue
            lines.append(f"- `{metric_name}`")
            for row in metric_rows.itertuples(index=False):
                lines.append(
                    f"  `{row.comparison}`: mean={row.mean:.6f}, "
                    f"95% CI [{row.ci_lower:.6f}, {row.ci_upper:.6f}]"
                )
        lines.append("")

    lines.append("## Notes")
    lines.append("- For `log_loss` and `brier_score`, lower is better.")
    lines.append("- For `accuracy`, higher is better.")
    lines.append("- If a delta CI includes 0, the difference is not clearly separable at this resolution.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate uncertainty for held-out metrics with session-level bootstrap, "
            "using fitted pairwise models from the task-interaction feature pipeline."
        )
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--split-dir",
        default="results/train_val_test_evaluation",
        help="Directory containing train/validation/test split parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/metric_uncertainty",
        help="Directory for bootstrap outputs.",
    )
    parser.add_argument(
        "--dataset-id",
        default="lmarena-ai/arena-human-preference-140k",
        help="HF dataset id used if formatting cache cannot be built from local raw data.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Number of session-level bootstrap replications per split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=1200,
        help="Maximum L-BFGS iterations per model fit.",
    )
    parser.add_argument(
        "--session-col",
        default="evaluation_session_id",
        help="Session/group column used as the bootstrap unit.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    split_dir = (repo_root / args.split_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = ["baseline", "full_formatting", "signal_interactions"]
    compare_pairs = [
        ("full_formatting", "baseline", "full_formatting_minus_baseline"),
        ("signal_interactions", "full_formatting", "signal_interactions_minus_full_formatting"),
    ]

    spec_map = _spec_map()
    missing = [name for name in model_names if name not in spec_map]
    if missing:
        raise ValueError(f"Requested model(s) missing from MODEL_SPECS: {missing}")

    print("Loading and preparing splits...")
    splits = load_split_parquets(split_dir)
    splits, transform_stats = ensure_proxy_features(splits)
    splits = ensure_formatting_feature(
        splits=splits,
        repo_root=repo_root,
        dataset_id=args.dataset_id,
        cache_path=output_dir / "formatting_features_cache.parquet",
    )
    splits = add_signal_interaction_features(splits)

    print("Fitting models on train split...")
    fitted: dict[str, object] = {}
    for model_name in model_names:
        fitted[model_name] = fit_pairwise_logit(
            splits["train"],
            feature_columns=spec_map[model_name],
            maxiter=args.maxiter,
        )

    rng = np.random.default_rng(args.seed)
    model_boot_rows: list[dict[str, object]] = []
    delta_boot_rows: list[dict[str, object]] = []

    print("Running session-level bootstrap on validation/test...")
    for split_name in ["validation", "test"]:
        frame = splits[split_name].copy()
        frame = frame.loc[frame["winner_binary"].notna()].copy()
        session_index = (
            frame[args.session_col].astype(str).drop_duplicates().sort_values().reset_index(drop=True)
        )
        n_sessions = len(session_index)
        if n_sessions == 0:
            raise ValueError(f"No sessions found for split '{split_name}'.")

        model_arrays: dict[str, dict[str, np.ndarray]] = {}
        for model_name in model_names:
            grouped = _session_metric_table(frame, fitted[model_name], args.session_col)
            model_arrays[model_name] = _align_session_arrays(grouped, session_index)

        for rep in range(args.n_bootstrap):
            idx = rng.integers(0, n_sessions, size=n_sessions, endpoint=False)
            rep_metrics: dict[str, dict[str, float]] = {}
            for model_name in model_names:
                metrics = _metrics_from_sample(model_arrays[model_name], idx)
                rep_metrics[model_name] = metrics
                for metric_name, value in metrics.items():
                    model_boot_rows.append(
                        {
                            "split": split_name,
                            "rep": rep,
                            "model_name": model_name,
                            "metric": metric_name,
                            "value": value,
                        }
                    )

            for model_a, model_b, label in compare_pairs:
                for metric_name in ["log_loss", "accuracy", "brier_score"]:
                    delta_boot_rows.append(
                        {
                            "split": split_name,
                            "rep": rep,
                            "comparison": label,
                            "model_a": model_a,
                            "model_b": model_b,
                            "metric": metric_name,
                            "value": rep_metrics[model_a][metric_name] - rep_metrics[model_b][metric_name],
                        }
                    )

    model_boot = pd.DataFrame(model_boot_rows)
    delta_boot = pd.DataFrame(delta_boot_rows)

    model_ci_rows: list[dict[str, object]] = []
    for (split_name, model_name, metric_name), subset in model_boot.groupby(
        ["split", "model_name", "metric"],
        sort=False,
    ):
        summary = _summarize_samples(subset["value"])
        model_ci_rows.append(
            {
                "split": split_name,
                "model_name": model_name,
                "metric": metric_name,
                **summary,
            }
        )
    model_ci = pd.DataFrame(model_ci_rows).sort_values(["split", "metric", "model_name"]).reset_index(drop=True)

    delta_ci_rows: list[dict[str, object]] = []
    for (split_name, comparison, model_a, model_b, metric_name), subset in delta_boot.groupby(
        ["split", "comparison", "model_a", "model_b", "metric"],
        sort=False,
    ):
        summary = _summarize_samples(subset["value"])
        delta_ci_rows.append(
            {
                "split": split_name,
                "comparison": comparison,
                "model_a": model_a,
                "model_b": model_b,
                "metric": metric_name,
                **summary,
            }
        )
    delta_ci = pd.DataFrame(delta_ci_rows).sort_values(["split", "metric", "comparison"]).reset_index(drop=True)

    model_boot.to_csv(output_dir / "bootstrap_model_metric_samples.csv", index=False)
    delta_boot.to_csv(output_dir / "bootstrap_delta_samples.csv", index=False)
    model_ci.to_csv(output_dir / "bootstrap_model_metric_summary.csv", index=False)
    delta_ci.to_csv(output_dir / "bootstrap_delta_summary.csv", index=False)
    pd.DataFrame([transform_stats]).to_csv(output_dir / "proxy_transform_stats.csv", index=False)

    summary_md = build_summary_markdown(
        model_ci=model_ci,
        delta_ci=delta_ci,
        n_bootstrap=args.n_bootstrap,
        compared_models=model_names,
    )
    (output_dir / "uncertainty_summary.md").write_text(summary_md, encoding="utf-8")

    print(f"Wrote: {output_dir / 'bootstrap_model_metric_samples.csv'}")
    print(f"Wrote: {output_dir / 'bootstrap_delta_samples.csv'}")
    print(f"Wrote: {output_dir / 'bootstrap_model_metric_summary.csv'}")
    print(f"Wrote: {output_dir / 'bootstrap_delta_summary.csv'}")
    print(f"Wrote: {output_dir / 'uncertainty_summary.md'}")


if __name__ == "__main__":
    main()
