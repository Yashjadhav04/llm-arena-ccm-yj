"""Fit formatting-augmented pairwise preference model.

Compares three models:
  baseline          — model skill only (no covariates)
  proxy_augmented   — + side_a_indicator + relative_response_length_z
  formatting_augmented — proxy_augmented + fmt_diff_has_markdown

Raw conversation text is loaded from data/raw/*.parquet when available;
otherwise streamed from HuggingFace to extract markdown formatting cues.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from src.models.pairwise_preference import evaluate_pairwise_logit, fit_pairwise_logit
from src.utils.arena_dataset import add_formatting_features


FORMATTING_FEATURE = "fmt_diff_has_markdown"

MODEL_SPECS = [
    {"model_name": "baseline", "feature_columns": []},
    {
        "model_name": "proxy_augmented",
        "feature_columns": ["side_a_indicator", "relative_response_length_z"],
    },
    {
        "model_name": "formatting_augmented",
        "feature_columns": [
            "side_a_indicator",
            "relative_response_length_z",
            FORMATTING_FEATURE,
        ],
    },
]

_TEXT_COLS = ["id", "conversation_a", "conversation_b"]


def _load_raw_text_local(repo_root: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(repo_root / "data" / "raw" / "*.parquet")))
    if not files:
        raise FileNotFoundError("No local parquet files under data/raw/")
    frames = [pd.read_parquet(p, columns=_TEXT_COLS) for p in files]
    df = pd.concat(frames, ignore_index=True)
    print(f"  Loaded local raw data: {len(df):,} rows from {len(files)} file(s)")
    return df


def _stream_raw_text_huggingface(
    dataset_id: str,
    needed_ids: set[str],
) -> pd.DataFrame:
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset(dataset_id, split="train", streaming=True)
    rows: list[dict] = []
    found: set[str] = set()
    for row in ds:
        row_id = row.get("id", "")
        if row_id in needed_ids and row_id not in found:
            rows.append(
                {
                    "id": row_id,
                    "conversation_a": row.get("conversation_a", []),
                    "conversation_b": row.get("conversation_b", []),
                }
            )
            found.add(row_id)
            if len(found) % 10_000 == 0:
                print(f"  ...streamed {len(found):,} / {len(needed_ids):,} rows")
            if found == needed_ids:
                break

    n_missed = len(needed_ids) - len(found)
    if n_missed:
        print(f"  Warning: {n_missed} split IDs not found in the raw dataset")
    print(f"  Streamed {len(rows):,} rows from {dataset_id}")
    return pd.DataFrame(rows)


def load_formatting_features(
    repo_root: Path,
    needed_ids: set[str],
    dataset_id: str,
) -> pd.DataFrame:
    """Return a DataFrame with columns [id, fmt_diff_has_markdown]."""
    try:
        raw = _load_raw_text_local(repo_root)
        raw = raw[raw["id"].isin(needed_ids)].reset_index(drop=True)
    except FileNotFoundError:
        print(f"  No local raw data — streaming from {dataset_id} ...")
        raw = _stream_raw_text_huggingface(dataset_id, needed_ids)

    flat = pd.DataFrame({"id": raw["id"].values})
    flat_fmt = add_formatting_features(flat, raw)
    return flat_fmt[["id", FORMATTING_FEATURE]].copy()


def fit_and_evaluate(
    splits: dict[str, pd.DataFrame],
    maxiter: int,
) -> tuple[pd.DataFrame, dict]:
    metrics_rows: list[dict] = []
    artifacts: dict = {}
    train = splits["train"]

    for spec in MODEL_SPECS:
        name = spec["model_name"]
        feats = spec["feature_columns"]
        fitted = fit_pairwise_logit(train, feature_columns=feats, maxiter=maxiter)
        artifacts[name] = fitted
        print(f"  Fitted {name}")

        for split_name, frame in splits.items():
            m = evaluate_pairwise_logit(frame, fitted, unknown_policy="drop")
            metrics_rows.append({"model_name": name, "split": split_name, **m})

    return pd.DataFrame(metrics_rows), artifacts


def print_comparison_table(metrics: pd.DataFrame) -> None:
    for split_name in ["validation", "test"]:
        subset = (
            metrics.loc[metrics["split"] == split_name][
                ["model_name", "accuracy", "log_loss", "brier_score"]
            ]
            .reset_index(drop=True)
        )
        print(f"\n{'='*55}")
        print(f"  {split_name.upper()} SET")
        print(f"{'='*55}")
        print(subset.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit formatting-augmented pairwise preference model."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--split-dir",
        default="results/train_val_test_evaluation",
        help="Directory containing train/validation/test parquets.",
    )
    parser.add_argument("--output-dir", default="results/formatting_model")
    parser.add_argument(
        "--dataset-id", default="lmarena-ai/arena-human-preference-140k"
    )
    parser.add_argument(
        "--maxiter", type=int, default=1000, help="Max L-BFGS iterations per fit."
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    split_dir = (repo_root / args.split_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing split parquets (already have proxy features)
    print("Loading split parquets...")
    splits: dict[str, pd.DataFrame] = {}
    for name in ["train", "validation", "test"]:
        df = pd.read_parquet(split_dir / f"{name}.parquet")
        splits[name] = df
        print(f"  {name}: {len(df):,} rows")

    all_ids: set[str] = set(pd.concat([df["id"] for df in splits.values()]))
    print(f"  Total unique IDs: {len(all_ids):,}")

    # Compute formatting features from raw conversations
    print("\nLoading raw conversations to compute formatting features...")
    fmt_features = load_formatting_features(
        repo_root=repo_root,
        needed_ids=all_ids,
        dataset_id=args.dataset_id,
    )
    print(f"  Got formatting features for {len(fmt_features):,} rows")

    # Merge formatting features into each split
    print("\nMerging formatting features...")
    augmented: dict[str, pd.DataFrame] = {}
    for name, frame in splits.items():
        merged = frame.merge(fmt_features, on="id", how="left")
        n_missing = int(merged[FORMATTING_FEATURE].isna().sum())
        if n_missing:
            print(f"  {name}: {n_missing} rows missing formatting features (filling 0)")
            merged[FORMATTING_FEATURE] = merged[FORMATTING_FEATURE].fillna(0.0)
        augmented[name] = merged
        val_counts = merged[FORMATTING_FEATURE].value_counts().to_dict()
        print(f"  {name}: fmt_diff_has_markdown distribution = {val_counts}")

    # Fit all three models and evaluate
    print("\nFitting models on train split...")
    metrics, artifacts = fit_and_evaluate(augmented, args.maxiter)

    # Save results
    metrics.to_csv(output_dir / "evaluation_metrics.csv", index=False)
    for name, fitted in artifacts.items():
        fitted.model_scores.to_csv(output_dir / f"{name}_leaderboard.csv", index=False)
        if not fitted.coefficients.empty:
            fitted.coefficients.to_csv(
                output_dir / f"{name}_coefficients.csv", index=False
            )

    print(f"\nSaved results to: {output_dir}")

    # Print comparison table
    print_comparison_table(metrics)

    # Print formatting-augmented coefficients
    fmt_coeff = artifacts["formatting_augmented"].coefficients
    print(f"\nFormatting-augmented coefficients:")
    print(fmt_coeff.to_string(index=False))


if __name__ == "__main__":
    main()
