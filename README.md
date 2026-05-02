# llm-arena-ccm

Computational Cognitive Modelling project on whether Chatbot Arena-style human
preferences reflect latent model quality alone, or are also shaped by observable
judgment cues such as response length, side position, formatting, and task
context.

## Current Status

The repo now has a reproducible analysis path from fixed train/validation/test
splits through bias-augmented pairwise preference models, uncertainty estimates,
and simulation-based model validation.

The current strongest model for calibrated held-out fit is `full_formatting`:
it improves test log loss over the baseline (`0.6357` vs `0.6465`) and is
validated by the focused simulation run in
`results/final_model_validation_30reps/validation_summary.md`.

The newer `signal_interactions` model is useful as a task-context sensitivity
analysis. It slightly improves accuracy, but worsens test log loss relative to
`full_formatting`, so it should probably not replace `full_formatting` as the
main final model.

## Repository Map

- `src/models/pairwise_preference.py`: Bradley-Terry-style pairwise logistic
  model with optional covariates.
- `src/utils/arena_dataset.py`: Arena dataset loading, feature flattening,
  task buckets, and markdown formatting features.
- `src/analysis/prepare_evaluation_splits.py`: materializes the fixed
  session-level split and first baseline/proxy evaluation.
- `src/analysis/evaluate_preference_models.py`: main comparison of
  baseline, length/position, formatting, and task-signal models.
- `src/analysis/bootstrap_metric_uncertainty.py`: session-level bootstrap confidence
  intervals for held-out metrics.
- `src/analysis/validate_final_model.py`: focused synthetic recovery
  validation for the chosen final model.
- `src/analysis/explore_full_dataset.py` and
  `src/analysis/explore_other_task_bucket.py`: exploratory EDA scripts.
- `docs/`: dataset and construct-mapping notes for the write-up.
- `results/`: generated experiment outputs. Some historical result folders are
  tracked; smoke/rerun outputs should be treated as local working artifacts
  unless you decide to preserve them.
- `archived/`: superseded first-pass scripts and teammate-generated historical
  outputs kept out of the active analysis flow.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code first looks for local raw parquet shards under `data/raw/`, then falls
back to cached Hugging Face Arrow files for `lmarena-ai/arena-human-preference-140k`.

## Main Commands

See `src/analysis/README.md` for when to use each script.

Rebuild the fixed split outputs:

```bash
python -m src.analysis.prepare_evaluation_splits --repo-root . --materialize-split-parquets
```

Run the main model comparison:

```bash
python -m src.analysis.evaluate_preference_models --repo-root .
```

Estimate metric uncertainty:

```bash
python -m src.analysis.bootstrap_metric_uncertainty --repo-root .
```

Run focused final-model validation:

```bash
python -m src.analysis.validate_final_model --repo-root .
```

## Remaining Work

The core modeling workflow is in good shape. The main items left are write-up
and scope decisions:

- Use `full_formatting` as the final model unless you want accuracy to dominate
  log-loss/calibration.
- Present `signal_interactions` as a sensitivity analysis showing that task
  context changes implied slopes but does not improve calibrated held-out fit.
- Decide whether to run the optional LLM-as-a-Judge comparison from the original
  project plan.
- Decide which generated result folders should be committed versus archived or
  left local.
