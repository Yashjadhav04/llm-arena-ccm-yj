from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.arena_dataset import extract_first_user_prompt, load_arena_raw


OTHER_SIGNAL_COLS = [
    "criteria_complexity",
    "criteria_specificity",
    "criteria_real_world",
]

CREATIVE_LEXICON = [
    "story",
    "poem",
    "lyrics",
    "fiction",
    "character",
    "roleplay",
    "creative",
    "novel",
    "imagine",
]

FACTUAL_LEXICON = [
    "explain",
    "how to",
    "what is",
    "why",
    "compare",
    "difference",
    "calculate",
    "solve",
    "proof",
    "analyze",
    "analysis",
    "python",
    "sql",
    "code",
    "legal",
    "medical",
    "translate",
    "summary",
]

STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "you",
    "are",
    "can",
    "what",
    "how",
    "why",
    "when",
    "where",
    "from",
    "into",
    "about",
    "your",
    "have",
    "please",
    "would",
    "should",
    "could",
    "them",
    "they",
    "their",
    "just",
    "make",
    "like",
    "need",
    "help",
    "using",
    "than",
    "then",
    "also",
    "will",
    "any",
    "all",
    "not",
    "but",
    "out",
    "who",
    "which",
    "was",
    "were",
    "has",
    "had",
    "our",
    "its",
    "it",
    "a",
    "an",
    "to",
    "of",
    "in",
    "on",
    "by",
    "or",
    "if",
    "is",
    "be",
    "as",
    "at",
    "we",
    "i",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']{2,}")


def subtype_label(row: pd.Series) -> str:
    active: list[str] = []
    if bool(row["criteria_complexity"]):
        active.append("complexity")
    if bool(row["criteria_specificity"]):
        active.append("specificity")
    if bool(row["criteria_real_world"]):
        active.append("real_world")
    if not active:
        return "none"
    return "+".join(active)


def lexical_bucket(prompt: str) -> str:
    text = prompt.lower()
    creative_hit = any(phrase in text for phrase in CREATIVE_LEXICON)
    factual_hit = any(phrase in text for phrase in FACTUAL_LEXICON)
    if creative_hit and factual_hit:
        return "creative_and_factual_like"
    if creative_hit:
        return "creative_like"
    if factual_hit:
        return "factual_like"
    return "unclear"


def tokenize(text: str) -> list[str]:
    tokens = [m.group(0).lower() for m in TOKEN_RE.finditer(text)]
    return [t for t in tokens if t not in STOPWORDS]


def top_tokens(
    texts: pd.Series,
    top_k: int = 40,
    use_document_frequency: bool = False,
) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    for text in texts.fillna(""):
        toks = tokenize(str(text))
        if use_document_frequency:
            counts.update(set(toks))
        else:
            counts.update(toks)
    rows = [{"token": token, "count": int(count)} for token, count in counts.most_common(top_k)]
    return pd.DataFrame(rows)


def top_tokens_by_group(frame: pd.DataFrame, group_col: str, top_k: int = 25) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_value, subset in frame.groupby(group_col):
        counts: Counter[str] = Counter()
        for text in subset["prompt_text"].fillna(""):
            counts.update(tokenize(str(text)))
        for rank, (token, count) in enumerate(counts.most_common(top_k), start=1):
            rows.append(
                {
                    group_col: group_value,
                    "rank": rank,
                    "token": token,
                    "count": int(count),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explore what the 'other' task bucket contains and whether it suggests new task features."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--split-dir", default="results/train_val_test_evaluation")
    parser.add_argument("--output-dir", default="results/full data eda/other_task_type")
    parser.add_argument(
        "--dataset-id",
        default="lmarena-ai/arena-human-preference-140k",
        help="Used if only HF cache is available for raw conversations.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Number of example prompts to save for manual inspection.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    split_dir = (repo_root / args.split_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_frames: list[pd.DataFrame] = []
    for split_name in ["train", "validation", "test"]:
        path = split_dir / f"{split_name}.parquet"
        frame = pd.read_parquet(path).copy()
        frame["split"] = split_name
        split_frames.append(frame)
    full = pd.concat(split_frames, ignore_index=True)
    full["id"] = full["id"].astype(str)

    other = full.loc[full["task_bucket"] == "other"].copy()
    if other.empty:
        raise ValueError("No rows found for task_bucket == 'other'.")

    other["other_subtype"] = other.apply(subtype_label, axis=1)

    subtype_counts = (
        other["other_subtype"]
        .value_counts()
        .rename_axis("other_subtype")
        .reset_index(name="count")
    )
    subtype_counts["share_within_other"] = subtype_counts["count"] / len(other)
    subtype_counts.to_csv(output_dir / "other_subtype_counts.csv", index=False)

    subtype_by_split = (
        other.groupby(["split", "other_subtype"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    split_totals = other["split"].value_counts().to_dict()
    subtype_by_split["share_within_split_other"] = subtype_by_split.apply(
        lambda row: row["count"] / split_totals[row["split"]],
        axis=1,
    )
    subtype_by_split.to_csv(output_dir / "other_subtype_by_split.csv", index=False)

    ids = set(other["id"])
    raw, source = load_arena_raw(
        repo_root=repo_root,
        limit=None,
        dataset_id=args.dataset_id,
        prefer_local=True,
    )
    raw["id"] = raw["id"].astype(str)
    raw = raw.loc[raw["id"].isin(ids), ["id", "conversation_a", "conversation_b"]].copy()

    raw["prompt_a"] = raw["conversation_a"].apply(extract_first_user_prompt)
    raw["prompt_b"] = raw["conversation_b"].apply(extract_first_user_prompt)
    raw["prompt_text"] = np.where(raw["prompt_a"].str.len() > 0, raw["prompt_a"], raw["prompt_b"])
    prompts = raw[["id", "prompt_text"]].copy()

    other = other.merge(prompts, on="id", how="left")
    other["prompt_text"] = other["prompt_text"].fillna("")
    other["prompt_char_len"] = other["prompt_text"].str.len()
    other["prompt_token_len"] = other["prompt_text"].apply(lambda x: len(tokenize(str(x))))
    other["lexical_bucket"] = other["prompt_text"].apply(lexical_bucket)

    lexical_counts = (
        other["lexical_bucket"]
        .value_counts()
        .rename_axis("lexical_bucket")
        .reset_index(name="count")
    )
    lexical_counts["share_within_other"] = lexical_counts["count"] / len(other)
    lexical_counts.to_csv(output_dir / "other_lexical_bucket_counts.csv", index=False)

    lexical_by_subtype = (
        other.groupby(["other_subtype", "lexical_bucket"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    subtype_totals = other["other_subtype"].value_counts().to_dict()
    lexical_by_subtype["share_within_subtype"] = lexical_by_subtype.apply(
        lambda row: row["count"] / subtype_totals[row["other_subtype"]],
        axis=1,
    )
    lexical_by_subtype.to_csv(output_dir / "other_lexical_by_subtype.csv", index=False)

    top_overall_cf = top_tokens(other["prompt_text"], top_k=60, use_document_frequency=False)
    top_overall_df = top_tokens(other["prompt_text"], top_k=60, use_document_frequency=True)
    top_overall_cf.to_csv(output_dir / "other_top_tokens_overall_cf.csv", index=False)
    top_overall_df.to_csv(output_dir / "other_top_tokens_overall_df.csv", index=False)

    top_by_subtype = top_tokens_by_group(other, group_col="other_subtype", top_k=30)
    top_by_subtype.to_csv(output_dir / "other_top_tokens_by_subtype.csv", index=False)

    examples = (
        other[["id", "split", "other_subtype", "lexical_bucket", "prompt_text"]]
        .sample(n=min(args.sample_size, len(other)), random_state=args.seed)
        .sort_values(["other_subtype", "lexical_bucket", "id"])
        .reset_index(drop=True)
    )
    examples["prompt_excerpt"] = examples["prompt_text"].str.replace(r"\s+", " ", regex=True).str.slice(0, 280)
    examples.to_csv(
        output_dir / "other_prompt_examples.csv",
        index=False,
    )

    mean_lengths = (
        other.groupby("other_subtype", as_index=False)[["prompt_char_len", "prompt_token_len"]]
        .mean()
        .rename(
            columns={
                "prompt_char_len": "mean_prompt_char_len",
                "prompt_token_len": "mean_prompt_token_len",
            }
        )
    )
    mean_lengths.to_csv(output_dir / "other_prompt_length_by_subtype.csv", index=False)

    lines: list[str] = []
    lines.append("# Other Bucket Exploration")
    lines.append("")
    lines.append("## Data Scope")
    lines.append(f"- Source for split data: `{split_dir}`")
    lines.append(f"- Source for raw prompt extraction: `{source}`")
    lines.append(f"- Total rows: {len(full):,}")
    lines.append(f"- `other` rows: {len(other):,} ({len(other)/len(full):.2%})")
    lines.append("")

    lines.append("## Subtype Breakdown Inside `other`")
    for row in subtype_counts.itertuples(index=False):
        lines.append(f"- `{row.other_subtype}`: n={int(row.count):,}, share={row.share_within_other:.2%}")
    lines.append("")

    lines.append("## Lexical Prompt Heuristic Inside `other`")
    for row in lexical_counts.itertuples(index=False):
        lines.append(f"- `{row.lexical_bucket}`: n={int(row.count):,}, share={row.share_within_other:.2%}")
    lines.append("")

    lines.append("## Top Prompt Tokens (Document Frequency)")
    for row in top_overall_df.head(20).itertuples(index=False):
        lines.append(f"- `{row.token}`: {int(row.count):,}")
    lines.append("")

    lines.append("## Recommendation")
    lines.append(
        "- Keep `creative` vs `factual_reasoning` as the main split; do not relabel `other` rows into those classes automatically."
    )
    lines.append(
        "- Use `criteria_complexity`, `criteria_specificity`, and `criteria_real_world` as orthogonal covariates (or interaction features) rather than forcing a new exclusive task label."
    )
    lines.append(
        "- If you want one additional task type, define a narrow `pragmatic_other` type from these criteria and test it as a sensitivity analysis."
    )
    lines.append("")

    summary_path = output_dir / "other_bucket_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote subtype counts: {output_dir / 'other_subtype_counts.csv'}")
    print(f"Wrote lexical counts: {output_dir / 'other_lexical_bucket_counts.csv'}")
    print(f"Wrote token summaries: {output_dir / 'other_top_tokens_overall_df.csv'}")
    print(f"Wrote prompt examples: {output_dir / 'other_prompt_examples.csv'}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
