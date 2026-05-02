# Final Model Validation Summary

## Configuration
- Final generator model: `full_formatting`
- Replications: 30
- Model selection split: `validation`
- Model selection metric: `log_loss` (lower is better)
- Train length scaling: mean=-0.4466, std=1637.5201

## Parameter Recovery
- Skill correlation: 0.9908 ± 0.0033
- Skill RMSE: 0.0545 ± 0.0098
- Coefficient correlation: 0.9981 ± 0.0017
- Coefficient RMSE: 0.0057 ± 0.0031

## Model Recovery (Selection Rates)
- `full_formatting` selected in 22/30 reps (0.733)
- `full_formatting_plus_signals` selected in 8/30 reps (0.267)

## Candidate Held-out Means
- `full_formatting`: log_loss=0.6374, accuracy=0.6341, brier_score=0.2235
- `full_formatting_plus_signals`: log_loss=0.6374, accuracy=0.6341, brier_score=0.2235
- `signal_interactions`: log_loss=0.6413, accuracy=0.6292, brier_score=0.2252
- `baseline`: log_loss=0.6453, accuracy=0.6250, brier_score=0.2270

## Notes
- This run validates only the final generator model, not a full confusion matrix across all generators.
- Strong validation evidence is: high parameter recovery and high self-selection rate for the final model.
