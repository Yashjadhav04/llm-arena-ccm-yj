"""Fit formatting-augmented pairwise preference models.

No existing files are modified. Raw conversation text is streamed from
HuggingFace (or read from data/raw/*.parquet if available) to compute
markdown formatting features via add_formatting_features(). Computed
features are cached to results/with_formatting/formatting_features_cache.parquet
so subsequent runs skip the stream. Results are written to results/with_formatting/.

Four models are compared:
  baseline             — model skill only
  proxy_augmented      — + side_a_indicator + relative_response_length_z
  formatting_augmented — proxy_augmented + fmt_diff_has_markdown (binary)
  full_formatting      — proxy_augmented + fmt_diff_n_{headers,bullets,bold,code_blocks}
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from src.models.pairwise_preference import evaluate_pairwise_logit, fit_pairwise_logit
from src.utils.arena_dataset import add_formatting_features


# ── model specs ────────────────────────────────────────────────────────────────

_PROXY_FEATS = ["side_a_indicator", "relative_response_length_z"]
_FMT_BINARY_FEAT = "fmt_diff_has_markdown"
_FMT_COUNT_FEATS = [
    "fmt_diff_n_headers",
    "fmt_diff_n_bullets",
    "fmt_diff_n_bold",
    "fmt_diff_n_code_blocks",
]
# All formatting columns we need to fetch / cache
ALL_FMT_COLS = [_FMT_BINARY_FEAT] + _FMT_COUNT_FEATS

MODEL_SPECS: list[dict] = [
    {
        "model_name": "baseline",
        "feature_columns": [],
    },
    {
        "model_name": "proxy_augmented",
        "feature_columns": _PROXY_FEATS,
    },
    {
        "model_name": "formatting_augmented",
        "feature_columns": _PROXY_FEATS + [_FMT_BINARY_FEAT],
    },
    {
        "model_name": "full_formatting",
        "feature_columns": _PROXY_FEATS + _FMT_COUNT_FEATS,
    },
]

_RAW_TEXT_COLS = ["id", "conversation_a", "conversation_b"]


# ── raw-text loading ────────────────────────────────────────────────────────────

def _local_raw_text(repo_root: Path) -> pd.DataFrame:
    """Read id + conversations from data/raw/*.parquet (raises if absent)."""
    files = sorted(glob.glob(str(repo_root / "data" / "raw" / "*.parquet")))
    if not files:
        raise FileNotFoundError("No parquet files in data/raw/")
    frames = [pd.read_parquet(p, columns=_RAW_TEXT_COLS) for p in files]
    df = pd.concat(frames, ignore_index=True)
    print(f"  Local raw data: {len(df):,} rows from {len(files)} file(s)")
    return df


def _stream_huggingface(dataset_id: str, needed_ids: set[str]) -> pd.DataFrame:
    """Stream only the rows we need from HuggingFace."""
    from datasets import load_dataset  # type: ignore[import]

    rows: list[dict] = []
    found: set[str] = set()
    ds = load_dataset(dataset_id, split="train", streaming=True)
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
                print(f"    ...collected {len(found):,} / {len(needed_ids):,} rows")
            if found == needed_ids:
                break

    n_missed = len(needed_ids) - len(found)
    if n_missed:
        print(f"  Warning: {n_missed} IDs not found in the raw dataset")
    print(f"  Streamed {len(rows):,} rows from {dataset_id}")
    return pd.DataFrame(rows)


def _extract_text(content: object) -> str:
    """Flatten the HuggingFace nested content format to a plain string.

    The dataset stores message content as a list of typed blocks, e.g.
      [{'type': 'text', 'text': '...', 'image': None, 'mimeType': None}]
    rather than a bare string.  _count_markdown_features expects a string,
    so we join all 'text' blocks here before handing off to arena_dataset.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def _normalize_conversations(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of raw where each message's content is a plain string."""
    raw = raw.copy()
    for col in ("conversation_a", "conversation_b"):
        if col not in raw.columns:
            continue
        raw[col] = raw[col].apply(
            lambda conv: (
                [
                    {**msg, "content": _extract_text(msg.get("content", ""))}
                    if isinstance(msg, dict)
                    else msg
                    for msg in conv
                ]
                if isinstance(conv, list)
                else conv
            )
        )
    return raw


def fetch_formatting_features(
    repo_root: Path,
    needed_ids: set[str],
    dataset_id: str,
    cache_path: Path,
) -> pd.DataFrame:
    """Return DataFrame with columns [id] + ALL_FMT_COLS.

    Loads from cache_path if it exists and covers all needed_ids; otherwise
    streams raw conversations, computes features, and writes the cache.
    """
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if needed_ids.issubset(set(cached["id"])):
            print(f"  Loaded formatting features from cache ({len(cached):,} rows)")
            return cached[["id"] + ALL_FMT_COLS].copy()
        print(f"  Cache exists but is incomplete — recomputing ...")

    try:
        raw = _local_raw_text(repo_root)
        raw = raw[raw["id"].isin(needed_ids)].reset_index(drop=True)
    except FileNotFoundError:
        print(f"  data/raw/ not found — streaming from {dataset_id} ...")
        raw = _stream_huggingface(dataset_id, needed_ids)

    # Flatten nested content blocks to plain strings before calling arena_dataset
    raw = _normalize_conversations(raw)

    # add_formatting_features does positional concat; flat just needs an id column
    flat_stub = pd.DataFrame({"id": raw["id"].values})
    enriched = add_formatting_features(flat_stub, raw)

    result = enriched[["id"] + ALL_FMT_COLS].copy()
    result.to_parquet(cache_path, index=False)
    print(f"  Cached formatting features to {cache_path.name}")
    return result


# ── model fitting ───────────────────────────────────────────────────────────────

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


# ── display ─────────────────────────────────────────────────────────────────────

def print_comparison_table(metrics: pd.DataFrame) -> None:
    for split_name in ["validation", "test"]:
        subset = (
            metrics.loc[metrics["split"] == split_name][
                ["model_name", "accuracy", "log_loss", "brier_score"]
            ]
            .reset_index(drop=True)
        )
        print(f"\n{'='*58}")
        print(f"  {split_name.upper()} SET")
        print(f"{'='*58}")
        print(subset.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


# ── entry point ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a formatting-augmented pairwise preference model."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root (default: current directory).",
    )
    parser.add_argument(
        "--split-dir",
        default="results/train_val_test_evaluation",
        help="Directory containing train/validation/test parquets.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/with_formatting",
        help="Where to write outputs (created if absent).",
    )
    parser.add_argument(
        "--dataset-id",
        default="lmarena-ai/arena-human-preference-140k",
        help="HuggingFace dataset ID used when no local raw parquets exist.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=1000,
        help="Max L-BFGS iterations per model fit.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    split_dir = (repo_root / args.split_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load existing splits (already have proxy features) ──────────────────
    print("Loading split parquets...")
    splits: dict[str, pd.DataFrame] = {}
    for name in ["train", "validation", "test"]:
        df = pd.read_parquet(split_dir / f"{name}.parquet")
        splits[name] = df
        print(f"  {name}: {len(df):,} rows, {len(df.columns)} cols")

    all_ids: set[str] = set(pd.concat([df["id"] for df in splits.values()]))
    print(f"  Total unique IDs across splits: {len(all_ids):,}")

    # ── 2. Fetch / cache formatting features ──────────────────────────────────
    cache_path = output_dir / "formatting_features_cache.parquet"
    print("\nFetching formatting features...")
    fmt_df = fetch_formatting_features(
        repo_root=repo_root,
        needed_ids=all_ids,
        dataset_id=args.dataset_id,
        cache_path=cache_path,
    )
    print(f"  Formatting features ready: {len(fmt_df):,} rows, {len(ALL_FMT_COLS)} feature cols")

    # ── 3. Merge all formatting columns into each split ────────────────────────
    print("\nMerging formatting features into splits...")
    augmented: dict[str, pd.DataFrame] = {}
    for name, frame in splits.items():
        merged = frame.merge(fmt_df, on="id", how="left")
        for col in ALL_FMT_COLS:
            n_missing = int(merged[col].isna().sum())
            if n_missing:
                print(f"  {name}/{col}: {n_missing} rows missing — filling 0")
                merged[col] = merged[col].fillna(0.0)
        augmented[name] = merged
    print("  Merge complete")

    # ── 4. Fit all four models ─────────────────────────────────────────────────
    print("\nFitting models on train split...")
    metrics, artifacts = fit_and_evaluate(augmented, args.maxiter)

    # ── 5. Save results ────────────────────────────────────────────────────────
    metrics.to_csv(output_dir / "evaluation_metrics.csv", index=False)

    for name, fitted in artifacts.items():
        fitted.model_scores.to_csv(output_dir / f"{name}_leaderboard.csv", index=False)
        if not fitted.coefficients.empty:
            fitted.coefficients.to_csv(
                output_dir / f"{name}_coefficients.csv", index=False
            )

    print(f"\nSaved to: {output_dir}")

    # ── 6. Display results ─────────────────────────────────────────────────────
    print_comparison_table(metrics)

    print("\nfull_formatting coefficients:")
    print(artifacts["full_formatting"].coefficients.to_string(index=False))


if __name__ == "__main__":
    main()
