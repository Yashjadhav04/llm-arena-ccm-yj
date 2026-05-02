# Analysis Scripts

Run scripts from the repository root with `python -m src.analysis.<module>`.

## Main Reporting Flow

1. `prepare_evaluation_splits.py`
   - Use first, or rerun only when the processed dataset or split policy changes.
   - Creates the fixed train/validation/test split and baseline/proxy outputs in `results/train_val_test_evaluation/`.
   - Keep `results/train_val_test_evaluation/`: downstream scripts use its `train.parquet`, `validation.parquet`, and `test.parquet` files as fixed inputs.
   - Typical command:
     ```bash
     python -m src.analysis.prepare_evaluation_splits --repo-root . --materialize-split-parquets
     ```

2. `evaluate_preference_models.py`
   - Use after splits exist.
   - Fits the main model ladder: baseline, proxy, formatting, full formatting, and signal/task-context variants.
   - Writes the main reporting outputs to `results/task_interactions/`.
   - Typical command:
     ```bash
     python -m src.analysis.evaluate_preference_models --repo-root .
     ```

3. `bootstrap_metric_uncertainty.py`
   - Use after the main model comparison.
   - Re-fits the reporting models and estimates session-bootstrap confidence intervals for held-out metrics.
   - Writes uncertainty outputs to `results/metric_uncertainty/`.
   - Typical command:
     ```bash
     python -m src.analysis.bootstrap_metric_uncertainty --repo-root .
     ```

4. `validate_final_model.py`
   - Use for final reporting validation.
   - Runs synthetic recovery for the selected final model, currently `full_formatting`.
   - Writes focused validation outputs to `results/final_model_validation_30reps/`.
   - Typical command:
     ```bash
     python -m src.analysis.validate_final_model --repo-root .
     ```

## Optional Diagnostics

- `explore_full_dataset.py`
  - Broad EDA over the Arena dataset.
  - Use when checking dataset fields, distributions, and first-pass descriptive patterns.

- `explore_other_task_bucket.py`
  - Focused EDA for prompts in the `other` task bucket.
  - Use when revisiting task taxonomy or explaining why `other` is not force-reclassified.

## Archived Scripts

- `archived/initial_pass/`
  - Early baseline and first-5k exploratory scripts. These are no longer part of the active workflow.

- `archived/with_formatting/`
  - Teammate-generated formatting-only run and its script. It is preserved as historical work, but the active formatting comparison now lives in `evaluate_preference_models.py`.

- `archived/model_validation_matrix/`
  - Optional full generator-by-candidate validation matrix. This is broader than the final paper claim and was not used for final reporting.
