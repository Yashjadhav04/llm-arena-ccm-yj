from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAW_COLUMNS = [
    "id",
    "model_a",
    "model_b",
    "winner",
    "evaluation_session_id",
    "evaluation_order",
    "conv_metadata",
    "category_tag",
    "language",
    "is_code",
    "timestamp",
    "conversation_a",
    "conversation_b",
]


def _safe_get(mapping: Any, *path: str, default: Any = None) -> Any:
    value = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _slice_split(limit: int | None) -> str:
    if limit is None:
        return "train"
    return f"train[:{int(limit)}]"


def load_arena_raw(
    repo_root: str | Path,
    limit: int | None = None,
    dataset_id: str = "lmarena-ai/arena-human-preference-140k",
    prefer_local: bool = True,
) -> tuple[pd.DataFrame, str]:
    repo_root = Path(repo_root)
    local_files = sorted(glob.glob(str(repo_root / "data" / "raw" / "*.parquet")))

    if prefer_local and local_files:
        frame = pd.concat(
            [pd.read_parquet(path, columns=RAW_COLUMNS) for path in local_files],
            ignore_index=True,
        )
        if limit is not None:
            frame = frame.head(limit).copy()
        return frame, f"local parquet files ({len(local_files)} file(s))"

    cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "lmarena-ai___parquet"
    cache_files = sorted(glob.glob(str(cache_root / "**" / "*.arrow"), recursive=True))
    if cache_files:
        from datasets import Dataset

        frames: list[pd.DataFrame] = []
        remaining = limit
        for path in cache_files:
            dataset = Dataset.from_file(path)
            if remaining is not None and remaining <= 0:
                break
            if remaining is not None and remaining < len(dataset):
                dataset = dataset.select(range(remaining))

            frame = dataset.to_pandas()[RAW_COLUMNS].copy()
            frames.append(frame)

            if remaining is not None:
                remaining -= len(frame)

        combined = pd.concat(frames, ignore_index=True)
        split = _slice_split(limit)
        source = f"{dataset_id} [{split}, cache arrow]"
        return combined, source

    raise FileNotFoundError(
        "No local parquet files were found under data/raw/, and no cached Hugging Face "
        "Arrow shards were available. Download the Arena dataset once or place parquet "
        "shards under data/raw/."
    )

# double check the actual domains through EDA -> precise buckets
def derive_task_bucket(row: pd.Series) -> str:
    creative_signal = bool(row["creative_writing"] or row["criteria_creativity"])
    factual_signal = bool(
        row["math"]
        or row["instruction_following"]
        or row["is_code"]
        or row["criteria_problem_solving"]
        or row["criteria_domain_knowledge"]
        or row["criteria_technical_accuracy"]
    )

    if creative_signal and factual_signal:
        return "mixed"
    if creative_signal:
        return "creative"
    if factual_signal:
        return "factual_reasoning"
    return "other"


def flatten_initial_features(raw: pd.DataFrame) -> pd.DataFrame:
    flat = pd.DataFrame(
        {
            "id": raw["id"],
            "model_a": raw["model_a"],
            "model_b": raw["model_b"],
            "winner": raw["winner"],
            "evaluation_session_id": raw["evaluation_session_id"],
            "evaluation_order": raw["evaluation_order"].fillna(0).astype(int),
            "language": raw["language"].fillna("unknown"),
            "is_code": raw["is_code"].fillna(False).astype(bool),
            "timestamp": pd.to_datetime(raw["timestamp"], errors="coerce"),
            "assistant_a_tokens": raw["conv_metadata"].apply(
                lambda x: _safe_get(x, "sum_assistant_a_tokens", default=np.nan)
            ),
            "assistant_b_tokens": raw["conv_metadata"].apply(
                lambda x: _safe_get(x, "sum_assistant_b_tokens", default=np.nan)
            ),
            "context_a_tokens": raw["conv_metadata"].apply(
                lambda x: _safe_get(x, "context_a_tokens", default=np.nan)
            ),
            "context_b_tokens": raw["conv_metadata"].apply(
                lambda x: _safe_get(x, "context_b_tokens", default=np.nan)
            ),
            "turns": raw["conv_metadata"].apply(
                lambda x: _safe_get(x, "turns", default=np.nan)
            ),
            "creative_writing": raw["category_tag"].apply(
                lambda x: bool(
                    _safe_get(
                        x,
                        "creative_writing_v0.1",
                        "creative_writing",
                        default=False,
                    )
                )
            ),
            "instruction_following": raw["category_tag"].apply(
                lambda x: bool(_safe_get(x, "if_v0.1", "if", default=False))
            ),
            "math": raw["category_tag"].apply(
                lambda x: bool(_safe_get(x, "math_v0.1", "math", default=False))
            ),
            "criteria_complexity": raw["category_tag"].apply(
                lambda x: bool(
                    _safe_get(x, "criteria_v0.1", "complexity", default=False)
                )
            ),
            "criteria_creativity": raw["category_tag"].apply(
                lambda x: bool(
                    _safe_get(x, "criteria_v0.1", "creativity", default=False)
                )
            ),
            "criteria_domain_knowledge": raw["category_tag"].apply(
                lambda x: bool(
                    _safe_get(
                        x, "criteria_v0.1", "domain_knowledge", default=False
                    )
                )
            ),
            "criteria_problem_solving": raw["category_tag"].apply(
                lambda x: bool(
                    _safe_get(x, "criteria_v0.1", "problem_solving", default=False)
                )
            ),
            "criteria_real_world": raw["category_tag"].apply(
                lambda x: bool(
                    _safe_get(x, "criteria_v0.1", "real_world", default=False)
                )
            ),
            "criteria_specificity": raw["category_tag"].apply(
                lambda x: bool(
                    _safe_get(x, "criteria_v0.1", "specificity", default=False)
                )
            ),
            "criteria_technical_accuracy": raw["category_tag"].apply(
                lambda x: bool(
                    _safe_get(
                        x, "criteria_v0.1", "technical_accuracy", default=False
                    )
                )
            ),
        }
    )

    flat["task_bucket"] = flat.apply(derive_task_bucket, axis=1)
    flat["winner_binary"] = np.where(
        flat["winner"].eq("model_a"),
        1.0,
        np.where(flat["winner"].eq("model_b"), 0.0, np.nan),
    )
    flat["is_binary_vote"] = flat["winner_binary"].notna()
    flat["length_diff_tokens"] = flat["assistant_a_tokens"] - flat["assistant_b_tokens"]
    flat["abs_length_diff_tokens"] = flat["length_diff_tokens"].abs()

    a_tokens = flat["assistant_a_tokens"].fillna(0) + 1
    b_tokens = flat["assistant_b_tokens"].fillna(0) + 1
    flat["log_length_ratio"] = np.log(a_tokens / b_tokens)

    mean_diff = flat["length_diff_tokens"].mean(skipna=True)
    std_diff = flat["length_diff_tokens"].std(skipna=True)
    if pd.isna(std_diff) or std_diff == 0:
        flat["length_diff_z"] = 0.0
    else:
        flat["length_diff_z"] = (flat["length_diff_tokens"] - mean_diff) / std_diff

    flat["side_a_bias"] = 1.0
    return flat


def _count_markdown_features(text: str) -> dict[str, int]:
    """Count surface markdown cues as proxies for a structure/fluency heuristic.

    These are observable surface statistics, not cognitive variables themselves.
    A voter who prefers a bulleted response may be responding to a latent
    structure heuristic; the bullet count is only the measurable correlate.
    """
    if not isinstance(text, str):
        return {"n_headers": 0, "n_bullets": 0, "n_bold": 0, "n_code_blocks": 0}
    return {
        "n_headers": text.count("\n#") + int(text.startswith("#")),
        "n_bullets": text.count("\n- ") + text.count("\n* "),
        "n_bold": text.count("**") // 2,
        "n_code_blocks": text.count("```") // 2,
    }


def add_formatting_features(flat: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Append markdown formatting features for responses A and B.

    Requires the raw frame to access conversation text. Adds per-response
    counts and a signed difference (A minus B) for each cue, which is the
    form used as a covariate in the pairwise logit model.
    """
    def extract_last_assistant_turn(conv: Any, role_key: str = "content") -> str:
        if not isinstance(conv, list):
            return ""
        assistant_turns = [
            m.get(role_key, "") for m in conv
            if isinstance(m, dict) and m.get("role") == "assistant"
        ]
        return assistant_turns[-1] if assistant_turns else ""

    texts_a = raw["conversation_a"].apply(extract_last_assistant_turn) if "conversation_a" in raw.columns else pd.Series([""] * len(raw))
    texts_b = raw["conversation_b"].apply(extract_last_assistant_turn) if "conversation_b" in raw.columns else pd.Series([""] * len(raw))

    feats_a = texts_a.apply(_count_markdown_features).apply(pd.Series).add_prefix("a_")
    feats_b = texts_b.apply(_count_markdown_features).apply(pd.Series).add_prefix("b_")

    flat = pd.concat([flat.reset_index(drop=True), feats_a.reset_index(drop=True), feats_b.reset_index(drop=True)], axis=1)

    for cue in ["n_headers", "n_bullets", "n_bold", "n_code_blocks"]:
        flat[f"fmt_diff_{cue}"] = flat[f"a_{cue}"] - flat[f"b_{cue}"]

    # Binary: did A use any markdown at all vs B?
    flat["a_has_markdown"] = (flat[["a_n_headers", "a_n_bullets", "a_n_bold", "a_n_code_blocks"]].sum(axis=1) > 0).astype(float)
    flat["b_has_markdown"] = (flat[["b_n_headers", "b_n_bullets", "b_n_bold", "b_n_code_blocks"]].sum(axis=1) > 0).astype(float)
    flat["fmt_diff_has_markdown"] = flat["a_has_markdown"] - flat["b_has_markdown"]

    return flat


def binary_vote_frame(flat: pd.DataFrame) -> pd.DataFrame:
    keep = flat["winner"].isin(["model_a", "model_b"])
    subset = flat.loc[keep].copy()
    subset["winner_binary"] = subset["winner_binary"].astype(float)
    return subset
