from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.models.pairwise_preference import fit_pairwise_logit
from src.utils.arena_dataset import binary_vote_frame, flatten_initial_features, load_arena_raw


def _format_top_counts(series: pd.Series, top_n: int = 5) -> list[str]:
    counts = series.value_counts(dropna=False).head(top_n)
    return [f"- `{idx}`: {int(value):,}" for idx, value in counts.items()]


def _task_specific_length_results(binary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bucket in ["creative", "factual_reasoning"]:
        subset = binary.loc[binary["task_bucket"] == bucket].copy()
        if len(subset) < 75:
            continue

        try:
            result = fit_pairwise_logit(
                subset,
                feature_columns=["side_a_bias", "length_diff_z"],
                maxiter=1000,
            )
        except Exception as exc:  # pragma: no cover - defensive for exploratory runs
            rows.append(
                {
                    "task_bucket": bucket,
                    "n_rows": len(subset),
                    "status": f"failed: {exc}",
                    "side_a_bias": None,
                    "length_diff_z": None,
                }
            )
            continue

        coeffs = result.coefficients.set_index("term")["estimate"]
        rows.append(
            {
                "task_bucket": bucket,
                "n_rows": len(subset),
                "status": "ok",
                "side_a_bias": float(coeffs.get("side_a_bias", 0.0)),
                "length_diff_z": float(coeffs.get("length_diff_z", 0.0)),
            }
        )

    return pd.DataFrame(rows)


def build_markdown_report(
    source: str,
    raw: pd.DataFrame,
    flat: pd.DataFrame,
    binary: pd.DataFrame,
    baseline,
    bias_model,
    task_results: pd.DataFrame,
) -> str:
    winner_series = raw["winner"].fillna("unknown").astype(str)
    task_series = flat["task_bucket"].fillna("unknown").astype(str)
    language_series = flat["language"].fillna("unknown").astype(str)

    baseline_top = baseline.model_scores.head(10)
    bias_top = bias_model.model_scores.head(10)
    bias_coeffs = bias_model.coefficients.set_index("term")["estimate"]

    lines: list[str] = []
    lines.append("# Immediate Next Steps Report")
    lines.append("")
    lines.append("## What This Run Covered")
    lines.append(f"- Data source: {source}")
    lines.append(f"- Raw rows loaded: {len(raw):,}")
    lines.append(f"- Binary rows used for the first models: {len(binary):,}")
    lines.append("- Immediate next steps executed:")
    lines.append("  - confirmed usable fields")
    lines.append("  - set an initial task split")
    lines.append("  - fit a baseline Bradley-Terry style model")
    lines.append("  - fit a first bias model with position and length terms")
    lines.append("")
    lines.append("## Confirmed First-Pass Fields")
    lines.append("- Outcome: `model_a`, `model_b`, `winner`")
    lines.append("- Session context: `evaluation_session_id`, `evaluation_order`, `language`, `timestamp`")
    lines.append("- Length features: `conv_metadata.sum_assistant_a_tokens`, `conv_metadata.sum_assistant_b_tokens`")
    lines.append("- Task split signals: `category_tag` subfields plus `is_code`")
    lines.append("")
    lines.append("## Recommended First Task Split")
    lines.append("- `creative`: creative writing or creativity signal")
    lines.append("- `factual_reasoning`: math, instruction following, code, problem solving, domain knowledge, or technical accuracy signal")
    lines.append("- `mixed`: both signal types")
    lines.append("- `other`: neither signal type")
    lines.append("")
    lines.append("### Task Bucket Counts")
    lines.extend(_format_top_counts(task_series, top_n=10))
    lines.append("")
    lines.append("### Winner Label Counts")
    lines.extend(_format_top_counts(winner_series, top_n=10))
    lines.append("")
    lines.append("### Top Languages")
    lines.extend(_format_top_counts(language_series, top_n=10))
    lines.append("")
    lines.append("## Baseline Model")
    lines.append(
        f"- Rows: {int(baseline.metrics['n_rows']):,}, models: {int(baseline.metrics['n_models'])}, "
        f"log loss: {baseline.metrics['log_loss']:.4f}, accuracy: {baseline.metrics['accuracy']:.4f}"
    )
    lines.append("- Top models by baseline score:")
    for row in baseline_top.itertuples(index=False):
        lines.append(f"  - `{row.model}`: {row.score:.4f}")
    lines.append("")
    lines.append("## First Bias Model")
    lines.append(
        f"- Rows: {int(bias_model.metrics['n_rows']):,}, models: {int(bias_model.metrics['n_models'])}, "
        f"log loss: {bias_model.metrics['log_loss']:.4f}, accuracy: {bias_model.metrics['accuracy']:.4f}"
    )
    lines.append(
        f"- Side-A position coefficient: {float(bias_coeffs.get('side_a_bias', 0.0)):.4f}"
    )
    lines.append(
        f"- Standardized length-difference coefficient: {float(bias_coeffs.get('length_diff_z', 0.0)):.4f}"
    )
    lines.append("- Top models by bias-adjusted score:")
    for row in bias_top.itertuples(index=False):
        lines.append(f"  - `{row.model}`: {row.score:.4f}")
    lines.append("")
    lines.append("## Task-Specific Bias Check")
    if task_results.empty:
        lines.append("- Not enough rows in the current run to fit separate creative and factual/reasoning models.")
    else:
        for row in task_results.itertuples(index=False):
            side_a = "n/a" if pd.isna(row.side_a_bias) else f"{row.side_a_bias:.4f}"
            length = "n/a" if pd.isna(row.length_diff_z) else f"{row.length_diff_z:.4f}"
            sample_note = " (small-sample)" if int(row.n_rows) < 150 else ""
            lines.append(
                f"- `{row.task_bucket}`{sample_note}: n={int(row.n_rows):,}, status={row.status}, "
                f"side_a_bias={side_a}, length_diff_z={length}"
            )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- The baseline model gives the first leaderboard estimate using model identity alone.")
    lines.append("- The bias model estimates whether side-A position and response length help explain votes beyond model identity.")
    lines.append("- The task split is ready for the next paper pass comparing creative versus factual/reasoning prompts.")
    lines.append("- These estimates are provisional because this run used the first 5,000 rows rather than the full 135,634-row dataset.")
    lines.append("")
    lines.append("## Recommended Next Actions")
    lines.append("- Extend the bias model with task interactions such as `length_diff_z * creative`.")
    lines.append("- Decide whether to keep only English rows or explicitly compare languages.")
    lines.append("- Add tie and `both_bad` handling after the binary comparison results are stable.")
    lines.append("- Start drafting the methods section around the baseline-versus-bias comparison.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute the first CCM project analysis steps.")
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root containing data/, src/, docs/, and results/ directories.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Optional row limit for a first-pass run. Ignored when full local parquet data are present unless explicitly small.",
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
        "--output-markdown",
        default="results/immediate_next_steps.md",
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--processed-parquet",
        default="data/processed/arena_initial_features.parquet",
        help="Processed first-pass feature table output path.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    raw, source = load_arena_raw(
        repo_root=repo_root,
        limit=args.limit,
        dataset_id=args.dataset_id,
        prefer_local=args.prefer_local,
    )
    flat = flatten_initial_features(raw)
    binary = binary_vote_frame(flat)

    processed_path = (repo_root / args.processed_parquet).resolve()
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    flat.to_parquet(processed_path, index=False)

    baseline = fit_pairwise_logit(binary)
    bias_model = fit_pairwise_logit(
        binary,
        feature_columns=["side_a_bias", "length_diff_z"],
    )
    task_results = _task_specific_length_results(binary)

    results_dir = (repo_root / "results").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    baseline.model_scores.to_csv(results_dir / "baseline_leaderboard.csv", index=False)
    bias_model.model_scores.to_csv(results_dir / "bias_leaderboard.csv", index=False)
    bias_model.coefficients.to_csv(results_dir / "bias_coefficients.csv", index=False)
    task_results.to_csv(results_dir / "task_bias_checks.csv", index=False)

    report = build_markdown_report(source, raw, flat, binary, baseline, bias_model, task_results)
    report_path = (repo_root / args.output_markdown).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(f"Wrote processed features to: {processed_path}")
    print(f"Wrote report to: {report_path}")
    print("Saved CSV outputs to results/.")


if __name__ == "__main__":
    main()
