from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.pairwise_preference import evaluate_pairwise_logit, fit_pairwise_logit
from src.utils.arena_dataset import add_formatting_features, load_arena_raw


FMT_BINARY_FEATURE = "fmt_diff_has_markdown"
FMT_COUNT_FEATURES = [
    "fmt_diff_n_headers",
    "fmt_diff_n_bullets",
    "fmt_diff_n_bold",
    "fmt_diff_n_code_blocks",
]
FMT_ALL_FEATURES = [FMT_BINARY_FEATURE, *FMT_COUNT_FEATURES]
TASK_ORDER = ["mixed", "factual_reasoning", "creative", "other"]
SIGNAL_STATE_ORDER = ["none", "creative_only", "factual_only", "both"]

MODEL_SPECS = [
    {"model_name": "baseline", "feature_columns": []},
    {
        "model_name": "proxy_augmented",
        "feature_columns": ["side_a_indicator", "relative_response_length_z"],
    },
    {
        "model_name": "formatting_augmented",
        "feature_columns": ["side_a_indicator", "relative_response_length_z", FMT_BINARY_FEATURE],
    },
    {
        "model_name": "full_formatting",
        "feature_columns": ["side_a_indicator", "relative_response_length_z", *FMT_COUNT_FEATURES],
    },
    {
        "model_name": "full_formatting_plus_signals",
        "feature_columns": [
            "side_a_indicator",
            "relative_response_length_z",
            *FMT_COUNT_FEATURES,
            "creative_signal",
            "factual_signal",
            "both_signal",
        ],
    },
    {
        "model_name": "signal_main_effects",
        "feature_columns": [
            "side_a_indicator",
            "relative_response_length_z",
            FMT_BINARY_FEATURE,
            "creative_signal",
            "factual_signal",
            "both_signal",
        ],
    },
    {
        "model_name": "signal_interactions",
        "feature_columns": [
            "side_a_indicator",
            "relative_response_length_z",
            FMT_BINARY_FEATURE,
            "creative_signal",
            "factual_signal",
            "both_signal",
            "len_x_creative",
            "len_x_factual",
            "len_x_both",
            "fmt_x_creative",
            "fmt_x_factual",
            "fmt_x_both",
        ],
    },
]


def load_split_parquets(split_dir: Path) -> dict[str, pd.DataFrame]:
    splits: dict[str, pd.DataFrame] = {}
    for split_name in ["train", "validation", "test"]:
        path = split_dir / f"{split_name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing split parquet: {path}")
        splits[split_name] = pd.read_parquet(path)
    return splits


def ensure_proxy_features(splits: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    prepared = {name: frame.copy() for name, frame in splits.items()}

    for frame in prepared.values():
        if "side_a_indicator" not in frame.columns:
            frame["side_a_indicator"] = 1.0

    train = prepared["train"]
    if "relative_response_length_z" not in train.columns:
        if "length_diff_tokens" not in train.columns:
            raise ValueError("Need either relative_response_length_z or length_diff_tokens.")

        mean_len = float(train["length_diff_tokens"].mean(skipna=True))
        std_len = float(train["length_diff_tokens"].std(skipna=True))

        for frame in prepared.values():
            if np.isnan(std_len) or std_len == 0:
                frame["relative_response_length_z"] = 0.0
            else:
                frame["relative_response_length_z"] = (frame["length_diff_tokens"] - mean_len) / std_len
    else:
        mean_len = float(train["length_diff_tokens"].mean(skipna=True))
        std_len = float(train["length_diff_tokens"].std(skipna=True))

    return prepared, {"train_length_mean": mean_len, "train_length_std": std_len}


def ensure_formatting_feature(
    splits: dict[str, pd.DataFrame],
    repo_root: Path,
    dataset_id: str,
    cache_path: Path,
) -> dict[str, pd.DataFrame]:
    if all(set(FMT_ALL_FEATURES).issubset(frame.columns) for frame in splits.values()):
        return {name: frame.copy() for name, frame in splits.items()}

    all_ids: set[str] = set(pd.concat([frame["id"].astype(str) for frame in splits.values()]))
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached["id"] = cached["id"].astype(str)
        cached_cols = set(cached.columns)
        if all_ids.issubset(set(cached["id"])) and set(FMT_ALL_FEATURES).issubset(cached_cols):
            cached_subset = cached[["id", *FMT_ALL_FEATURES]].copy()
            has_any_signal = bool((cached_subset[FMT_ALL_FEATURES] != 0).any().any())
            if has_any_signal:
                fmt_df = cached_subset
            else:
                print("  Cached formatting features are all zero; recomputing cache...")
                fmt_df = pd.DataFrame()
        else:
            fmt_df = pd.DataFrame()
    else:
        fmt_df = pd.DataFrame()

    if fmt_df.empty:
        raw, _source = load_arena_raw(
            repo_root=repo_root,
            limit=None,
            dataset_id=dataset_id,
            prefer_local=True,
        )
        raw["id"] = raw["id"].astype(str)
        raw = raw.loc[raw["id"].isin(all_ids), ["id", "conversation_a", "conversation_b"]].reset_index(drop=True)

        flat_stub = pd.DataFrame({"id": raw["id"].values})
        flat_fmt = add_formatting_features(flat_stub, raw)
        fmt_df = flat_fmt[["id", *FMT_ALL_FEATURES]].copy()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fmt_df.to_parquet(cache_path, index=False)

    merged: dict[str, pd.DataFrame] = {}
    for split_name, frame in splits.items():
        frame_copy = frame.copy()
        frame_copy["id"] = frame_copy["id"].astype(str)
        out = frame_copy.merge(fmt_df, on="id", how="left")
        for col in FMT_ALL_FEATURES:
            if out[col].isna().any():
                out[col] = out[col].fillna(0.0)
        merged[split_name] = out
    return merged


def add_signal_interaction_features(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    enriched: dict[str, pd.DataFrame] = {}
    for split_name, frame in splits.items():
        out = frame.copy()
        creative_signal = (
            out["creative_writing"].fillna(False).astype(bool)
            | out["criteria_creativity"].fillna(False).astype(bool)
        ).astype(float)
        factual_signal = (
            out["math"].fillna(False).astype(bool)
            | out["instruction_following"].fillna(False).astype(bool)
            | out["is_code"].fillna(False).astype(bool)
            | out["criteria_problem_solving"].fillna(False).astype(bool)
            | out["criteria_domain_knowledge"].fillna(False).astype(bool)
            | out["criteria_technical_accuracy"].fillna(False).astype(bool)
        ).astype(float)

        out["creative_signal"] = creative_signal
        out["factual_signal"] = factual_signal
        out["both_signal"] = out["creative_signal"] * out["factual_signal"]

        out["len_x_creative"] = out["relative_response_length_z"] * out["creative_signal"]
        out["len_x_factual"] = out["relative_response_length_z"] * out["factual_signal"]
        out["len_x_both"] = out["relative_response_length_z"] * out["both_signal"]

        out["fmt_x_creative"] = out[FMT_BINARY_FEATURE] * out["creative_signal"]
        out["fmt_x_factual"] = out[FMT_BINARY_FEATURE] * out["factual_signal"]
        out["fmt_x_both"] = out[FMT_BINARY_FEATURE] * out["both_signal"]

        out["signal_state"] = np.select(
            [
                (out["creative_signal"] == 0.0) & (out["factual_signal"] == 0.0),
                (out["creative_signal"] == 1.0) & (out["factual_signal"] == 0.0),
                (out["creative_signal"] == 0.0) & (out["factual_signal"] == 1.0),
                (out["creative_signal"] == 1.0) & (out["factual_signal"] == 1.0),
            ],
            SIGNAL_STATE_ORDER,
            default="none",
        )
        enriched[split_name] = out
    return enriched


def fit_and_evaluate_global(
    splits: dict[str, pd.DataFrame],
    maxiter: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    artifacts: dict[str, object] = {}
    train = splits["train"]

    for spec in MODEL_SPECS:
        model_name = spec["model_name"]
        feature_columns = spec["feature_columns"]
        fitted = fit_pairwise_logit(train, feature_columns=feature_columns, maxiter=maxiter)
        artifacts[model_name] = fitted

        for split_name, frame in splits.items():
            metrics = evaluate_pairwise_logit(frame, fitted, unknown_policy="drop")
            rows.append(
                {
                    "model_name": model_name,
                    "split": split_name,
                    **metrics,
                }
            )

    return pd.DataFrame(rows), artifacts


def evaluate_by_task(
    splits: dict[str, pd.DataFrame],
    artifacts: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, fitted in artifacts.items():
        for split_name in ["validation", "test"]:
            frame = splits[split_name]
            for task_bucket in TASK_ORDER:
                subset = frame.loc[frame["task_bucket"] == task_bucket].copy()
                if subset.empty:
                    continue
                metrics = evaluate_pairwise_logit(subset, fitted, unknown_policy="drop")
                rows.append(
                    {
                        "model_name": model_name,
                        "split": split_name,
                        "task_bucket": task_bucket,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def build_split_task_counts(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, frame in splits.items():
        counts = frame["task_bucket"].value_counts(dropna=False)
        total = len(frame)
        for task_bucket, count in counts.items():
            rows.append(
                {
                    "split": split_name,
                    "task_bucket": str(task_bucket),
                    "count": int(count),
                    "share": float(count / total),
                }
            )
    return pd.DataFrame(rows)


def evaluate_by_signal_state(
    splits: dict[str, pd.DataFrame],
    artifacts: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, fitted in artifacts.items():
        for split_name in ["validation", "test"]:
            frame = splits[split_name]
            for signal_state in SIGNAL_STATE_ORDER:
                subset = frame.loc[frame["signal_state"] == signal_state].copy()
                if subset.empty:
                    continue
                metrics = evaluate_pairwise_logit(subset, fitted, unknown_policy="drop")
                rows.append(
                    {
                        "model_name": model_name,
                        "split": split_name,
                        "signal_state": signal_state,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def build_split_signal_state_counts(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, frame in splits.items():
        counts = frame["signal_state"].value_counts(dropna=False)
        total = len(frame)
        for signal_state, count in counts.items():
            rows.append(
                {
                    "split": split_name,
                    "signal_state": str(signal_state),
                    "count": int(count),
                    "share": float(count / total),
                }
            )
    return pd.DataFrame(rows)


def derive_signal_effects(coefficients: pd.DataFrame) -> pd.DataFrame:
    coeff_map = coefficients.set_index("term")["estimate"].to_dict()

    base_len = float(coeff_map.get("relative_response_length_z", 0.0))
    base_fmt = float(coeff_map.get(FMT_BINARY_FEATURE, 0.0))

    rows = []
    for signal_state in SIGNAL_STATE_ORDER:
        if signal_state == "none":
            len_effect = base_len
            fmt_effect = base_fmt
        elif signal_state == "creative_only":
            len_effect = base_len + float(coeff_map.get("len_x_creative", 0.0))
            fmt_effect = base_fmt + float(coeff_map.get("fmt_x_creative", 0.0))
        elif signal_state == "factual_only":
            len_effect = base_len + float(coeff_map.get("len_x_factual", 0.0))
            fmt_effect = base_fmt + float(coeff_map.get("fmt_x_factual", 0.0))
        else:
            len_effect = (
                base_len
                + float(coeff_map.get("len_x_creative", 0.0))
                + float(coeff_map.get("len_x_factual", 0.0))
                + float(coeff_map.get("len_x_both", 0.0))
            )
            fmt_effect = (
                base_fmt
                + float(coeff_map.get("fmt_x_creative", 0.0))
                + float(coeff_map.get("fmt_x_factual", 0.0))
                + float(coeff_map.get("fmt_x_both", 0.0))
            )
        rows.append(
            {
                "signal_state": signal_state,
                "implied_length_effect": len_effect,
                "implied_formatting_effect": fmt_effect,
            }
        )
    return pd.DataFrame(rows)


def build_summary_markdown(
    global_metrics: pd.DataFrame,
    task_metrics: pd.DataFrame,
    split_task_counts: pd.DataFrame,
    signal_metrics: pd.DataFrame,
    split_signal_state_counts: pd.DataFrame,
    interaction_effects: pd.DataFrame,
) -> str:
    model_order = [spec["model_name"] for spec in MODEL_SPECS]
    available_models = [
        name for name in model_order if name in set(global_metrics["model_name"].astype(str))
    ]

    lines: list[str] = []
    lines.append("# Creative/Factual Signal Interaction Evaluation")
    lines.append("")
    lines.append("## Global Held-out Metrics")
    for split_name in ["validation", "test"]:
        lines.append(f"### {split_name.title()}")
        subset = global_metrics.loc[global_metrics["split"] == split_name].copy()
        subset["model_name"] = pd.Categorical(
            subset["model_name"],
            categories=available_models,
            ordered=True,
        )
        subset = subset.sort_values("model_name")
        for row in subset.itertuples(index=False):
            lines.append(
                f"- `{row.model_name}`: log_loss={row.log_loss:.4f}, accuracy={row.accuracy:.4f}, "
                f"brier_score={row.brier_score:.4f}"
            )
        lines.append("")

    comparison_model = None
    if "full_formatting" in available_models:
        comparison_model = "full_formatting"
    elif "formatting_augmented" in available_models:
        comparison_model = "formatting_augmented"

    target_model = "signal_interactions" if "signal_interactions" in available_models else None

    if comparison_model is not None and target_model is not None:
        val = global_metrics.pivot(index="split", columns="model_name", values="log_loss")
        acc = global_metrics.pivot(index="split", columns="model_name", values="accuracy")
        lines.append(
            "## Held-out Delta: "
            f"{target_model.replace('_', ' ').title()} Minus "
            f"{comparison_model.replace('_', ' ').title()}"
        )
        for split_name in ["validation", "test"]:
            lines.append(
                f"- `{split_name}` log_loss delta: "
                f"{(val.loc[split_name, target_model] - val.loc[split_name, comparison_model]):.4f}"
            )
            lines.append(
                f"- `{split_name}` accuracy delta: "
                f"{(acc.loc[split_name, target_model] - acc.loc[split_name, comparison_model]):.4f}"
            )
        lines.append("")

    lines.append("## Per-task Held-out Log Loss")
    for split_name in ["validation", "test"]:
        lines.append(f"### {split_name.title()}")
        subset = task_metrics.loc[task_metrics["split"] == split_name].copy()
        pivot = subset.pivot(index="task_bucket", columns="model_name", values="log_loss")
        for task_bucket in TASK_ORDER:
            if task_bucket not in pivot.index:
                continue
            parts: list[str] = []
            for model_name in available_models:
                if model_name not in pivot.columns:
                    continue
                value = pivot.loc[task_bucket, model_name]
                if pd.isna(value):
                    continue
                parts.append(f"{model_name}={value:.4f}")
            lines.append(f"- `{task_bucket}`: " + ", ".join(parts))
        lines.append("")

    lines.append("## Per-signal-state Held-out Log Loss")
    for split_name in ["validation", "test"]:
        lines.append(f"### {split_name.title()}")
        subset = signal_metrics.loc[signal_metrics["split"] == split_name].copy()
        pivot = subset.pivot(index="signal_state", columns="model_name", values="log_loss")
        for signal_state in SIGNAL_STATE_ORDER:
            if signal_state not in pivot.index:
                continue
            parts: list[str] = []
            for model_name in available_models:
                if model_name not in pivot.columns:
                    continue
                value = pivot.loc[signal_state, model_name]
                if pd.isna(value):
                    continue
                parts.append(f"{model_name}={value:.4f}")
            lines.append(f"- `{signal_state}`: " + ", ".join(parts))
        lines.append("")

    lines.append("## Split Task Counts")
    for split_name in ["train", "validation", "test"]:
        subset = split_task_counts.loc[split_task_counts["split"] == split_name]
        for row in subset.itertuples(index=False):
            lines.append(
                f"- `{split_name}` `{row.task_bucket}`: n={int(row.count):,}, share={row.share:.4f}"
            )
    lines.append("")

    lines.append("## Split Signal-State Counts")
    for split_name in ["train", "validation", "test"]:
        subset = split_signal_state_counts.loc[split_signal_state_counts["split"] == split_name]
        for row in subset.itertuples(index=False):
            lines.append(
                f"- `{split_name}` `{row.signal_state}`: n={int(row.count):,}, share={row.share:.4f}"
            )
    lines.append("")

    lines.append("## Signal-state Implied Slopes (Signal Interaction Model)")
    for row in interaction_effects.itertuples(index=False):
        lines.append(
            f"- `{row.signal_state}`: length_effect={row.implied_length_effect:.4f}, "
            f"formatting_effect={row.implied_formatting_effect:.4f}"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- `creative_signal` and `factual_signal` are non-exclusive and can both equal 1.")
    lines.append("- `both_signal = creative_signal * factual_signal` captures overlap effects.")
    lines.append("- Per-task metrics are held-out only (`validation`, `test`).")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run held-out creative/factual signal interaction evaluation for preference models.",
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--split-dir",
        default="results/train_val_test_evaluation",
        help="Directory with train/validation/test parquets.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/task_interactions",
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--dataset-id",
        default="lmarena-ai/arena-human-preference-140k",
        help="HF dataset ID used if raw local conversations are unavailable.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=1000,
        help="Maximum L-BFGS iterations per model fit.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    split_dir = (repo_root / args.split_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading split parquets...")
    splits = load_split_parquets(split_dir)
    for split_name, frame in splits.items():
        print(f"  {split_name}: {len(frame):,} rows")

    print("Ensuring proxy features...")
    splits, transform_stats = ensure_proxy_features(splits)

    print("Ensuring formatting feature...")
    cache_path = output_dir / "formatting_features_cache.parquet"
    splits = ensure_formatting_feature(
        splits=splits,
        repo_root=repo_root,
        dataset_id=args.dataset_id,
        cache_path=cache_path,
    )

    print("Adding creative/factual signal interaction columns...")
    splits = add_signal_interaction_features(splits)

    print("Fitting and evaluating models...")
    global_metrics, artifacts = fit_and_evaluate_global(splits, maxiter=args.maxiter)
    task_metrics = evaluate_by_task(splits, artifacts)
    signal_metrics = evaluate_by_signal_state(splits, artifacts)
    split_task_counts = build_split_task_counts(splits)
    split_signal_state_counts = build_split_signal_state_counts(splits)
    interaction_effects = derive_signal_effects(
        artifacts["signal_interactions"].coefficients
    )

    global_metrics.to_csv(output_dir / "evaluation_metrics_global.csv", index=False)
    task_metrics.to_csv(output_dir / "evaluation_metrics_by_task.csv", index=False)
    signal_metrics.to_csv(output_dir / "evaluation_metrics_by_signal_state.csv", index=False)
    split_task_counts.to_csv(output_dir / "split_task_counts.csv", index=False)
    split_signal_state_counts.to_csv(output_dir / "split_signal_state_counts.csv", index=False)
    pd.DataFrame([transform_stats]).to_csv(output_dir / "proxy_transform_stats.csv", index=False)
    interaction_effects.to_csv(output_dir / "signal_interaction_effects.csv", index=False)

    for model_name, fitted in artifacts.items():
        fitted.model_scores.to_csv(output_dir / f"{model_name}_leaderboard.csv", index=False)
        if not fitted.coefficients.empty:
            fitted.coefficients.to_csv(output_dir / f"{model_name}_coefficients.csv", index=False)

    summary = build_summary_markdown(
        global_metrics=global_metrics,
        task_metrics=task_metrics,
        split_task_counts=split_task_counts,
        signal_metrics=signal_metrics,
        split_signal_state_counts=split_signal_state_counts,
        interaction_effects=interaction_effects,
    )
    summary_path = output_dir / "evaluation_summary.md"
    summary_path.write_text(summary, encoding="utf-8")

    print(f"Wrote global metrics: {output_dir / 'evaluation_metrics_global.csv'}")
    print(f"Wrote per-task metrics: {output_dir / 'evaluation_metrics_by_task.csv'}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
