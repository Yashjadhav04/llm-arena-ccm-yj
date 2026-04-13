"""Reusable helpers for loading the fixed train/validation/test split.

Typical use in a new experiment script:

    from src.analysis.dataset_splits import load_fixed_binary_splits

    splits = load_fixed_binary_splits(repo_root=".")
    train, validation, test = splits.train, splits.validation, splits.test
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.arena_dataset import binary_vote_frame, flatten_initial_features, load_arena_raw


@dataclass
class FixedBinarySplits:
    source: str
    split_source: str
    frame: pd.DataFrame

    @property
    def train(self) -> pd.DataFrame:
        return self.frame.loc[self.frame["split"] == "train"].copy()

    @property
    def validation(self) -> pd.DataFrame:
        return self.frame.loc[self.frame["split"] == "validation"].copy()

    @property
    def test(self) -> pd.DataFrame:
        return self.frame.loc[self.frame["split"] == "test"].copy()

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


def load_feature_table(
    repo_root: str | Path,
    processed_parquet: str | Path = "data/processed/arena_full_features.parquet",
    limit: int | None = None,
    dataset_id: str = "lmarena-ai/arena-human-preference-140k",
    prefer_local: bool = True,
) -> tuple[pd.DataFrame, str]:
    repo_root = Path(repo_root).resolve()
    processed_parquet = (repo_root / processed_parquet).resolve()

    if processed_parquet.exists() and limit is None:
        frame = pd.read_parquet(processed_parquet)
        return frame, f"processed feature table ({processed_parquet.relative_to(repo_root)})"

    raw, source = load_arena_raw(
        repo_root=repo_root,
        limit=limit,
        dataset_id=dataset_id,
        prefer_local=prefer_local,
    )
    frame = flatten_initial_features(raw)
    processed_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(processed_parquet, index=False)
    return frame, source


def assign_session_splits(
    binary: pd.DataFrame,
    train_share: float,
    validation_share: float,
    seed: int,
    max_attempts: int = 50,
) -> pd.DataFrame:
    if train_share <= 0 or validation_share <= 0 or train_share + validation_share >= 1:
        raise ValueError("train_share and validation_share must leave room for a non-empty test split.")

    session_sizes = (
        binary.groupby("evaluation_session_id", dropna=False)
        .size()
        .rename("rows_in_session")
        .reset_index()
    )
    total_rows = float(session_sizes["rows_in_session"].sum())

    for attempt in range(max_attempts):
        shuffled = session_sizes.sample(frac=1.0, random_state=seed + attempt).reset_index(drop=True)
        cumulative_share = shuffled["rows_in_session"].cumsum() / total_rows
        shuffled["split"] = "test"
        shuffled.loc[cumulative_share <= train_share, "split"] = "train"
        validation_mask = (cumulative_share > train_share) & (
            cumulative_share <= train_share + validation_share
        )
        shuffled.loc[validation_mask, "split"] = "validation"

        split_map = shuffled.set_index("evaluation_session_id")["split"]
        assigned = binary.copy()
        assigned["split"] = assigned["evaluation_session_id"].map(split_map)

        if assigned["split"].isna().any():
            raise RuntimeError("Missing split assignment for some rows.")

        split_names = set(assigned["split"])
        if split_names != {"train", "validation", "test"}:
            continue

        train_models = set(assigned.loc[assigned["split"] == "train", "model_a"]).union(
            set(assigned.loc[assigned["split"] == "train", "model_b"])
        )
        validation_models = set(
            assigned.loc[assigned["split"] == "validation", "model_a"]
        ).union(set(assigned.loc[assigned["split"] == "validation", "model_b"]))
        test_models = set(assigned.loc[assigned["split"] == "test", "model_a"]).union(
            set(assigned.loc[assigned["split"] == "test", "model_b"])
        )

        if validation_models.issubset(train_models) and test_models.issubset(train_models):
            return assigned

    raise RuntimeError(
        "Unable to create a session-level train/validation/test split where validation "
        "and test models are all observed in training."
    )


def apply_saved_split_assignments(
    binary: pd.DataFrame,
    split_assignments_path: str | Path,
) -> pd.DataFrame:
    split_assignments_path = Path(split_assignments_path).expanduser().resolve()
    assignments = pd.read_csv(split_assignments_path)
    required = {"id", "split"}
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(
            f"Split assignment file {split_assignments_path} is missing required columns: {sorted(missing)}"
        )

    if assignments["id"].duplicated().any():
        raise ValueError(f"Split assignment file {split_assignments_path} contains duplicate ids.")

    merged = binary.merge(
        assignments[["id", "split"]],
        on="id",
        how="left",
        validate="one_to_one",
    )
    if merged["split"].isna().any():
        missing_ids = merged.loc[merged["split"].isna(), "id"].head(5).tolist()
        raise ValueError(
            f"Saved split assignments do not cover all binary rows. Example missing ids: {missing_ids}"
        )

    train_models = set(merged.loc[merged["split"] == "train", "model_a"]).union(
        set(merged.loc[merged["split"] == "train", "model_b"])
    )
    for split_name in ["validation", "test"]:
        split_models = set(merged.loc[merged["split"] == split_name, "model_a"]).union(
            set(merged.loc[merged["split"] == split_name, "model_b"])
        )
        unseen = split_models - train_models
        if unseen:
            unseen_preview = sorted(unseen)[:5]
            raise ValueError(
                f"Saved split {split_name} includes models not seen in train. "
                f"Examples: {unseen_preview}"
            )

    return merged


def load_fixed_binary_splits(
    repo_root: str | Path = ".",
    processed_parquet: str | Path = "data/processed/arena_full_features.parquet",
    split_assignments_path: str | Path = "results/train_val_test_evaluation/split_assignments.csv",
    limit: int | None = None,
    dataset_id: str = "lmarena-ai/arena-human-preference-140k",
    prefer_local: bool = True,
) -> FixedBinarySplits:
    flat, source = load_feature_table(
        repo_root=repo_root,
        processed_parquet=processed_parquet,
        limit=limit,
        dataset_id=dataset_id,
        prefer_local=prefer_local,
    )
    binary = binary_vote_frame(flat)
    split_frame = apply_saved_split_assignments(
        binary=binary,
        split_assignments_path=Path(repo_root).resolve() / split_assignments_path,
    )
    return FixedBinarySplits(
        source=source,
        split_source=str((Path(repo_root).resolve() / split_assignments_path).resolve()),
        frame=split_frame,
    )


def materialize_split_datasets(
    split_frame: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "validation", "test"]:
        subset = split_frame.loc[split_frame["split"] == split_name].copy()
        subset.to_parquet(output_dir / f"{split_name}.parquet", index=False)
