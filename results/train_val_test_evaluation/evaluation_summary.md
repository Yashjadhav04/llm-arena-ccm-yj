# Train / Validation / Test Evaluation

## Data Source
- processed feature table (data/processed/arena_full_features.parquet)

## Split Summary
- `train`: rows=78,678, row_share=0.8000, sessions=69,028, model_a_win_rate=0.4937
- `validation`: rows=9,835, row_share=0.1000, sessions=8,660, model_a_win_rate=0.4937
- `test`: rows=9,835, row_share=0.1000, sessions=8,638, model_a_win_rate=0.4946

## Proxy Feature Scaling
- Train-set length mean: -0.4466
- Train-set length std: 1637.5201

## Metrics By Model And Split
### Train
- `baseline`: log_loss=0.6453, accuracy=0.6313, brier_score=0.2269, covered_row_share=1.0000
- `proxy_augmented`: log_loss=0.6412, accuracy=0.6375, brier_score=0.2242, covered_row_share=1.0000

### Validation
- `baseline`: log_loss=0.6456, accuracy=0.6268, brier_score=0.2270, covered_row_share=1.0000
- `proxy_augmented`: log_loss=0.6399, accuracy=0.6377, brier_score=0.2241, covered_row_share=1.0000

### Test
- `baseline`: log_loss=0.6465, accuracy=0.6274, brier_score=0.2274, covered_row_share=1.0000
- `proxy_augmented`: log_loss=0.6415, accuracy=0.6362, brier_score=0.2245, covered_row_share=1.0000

## Validation Delta: Proxy Minus Baseline
- Log loss delta: -0.0057
- Accuracy delta: 0.0109
- Brier score delta: -0.0029

## Test Delta: Proxy Minus Baseline
- Log loss delta: -0.0051
- Accuracy delta: 0.0088
- Brier score delta: -0.0029

## Proxy Coefficients
- `side_a_indicator`: -0.0266
- `relative_response_length_z`: 0.2935

## Notes
- Splits are session-level to reduce leakage across related votes.
- Report test-set metrics as the main comparison in the write-up.
- Use validation metrics for model/feature iteration before touching the test set.
