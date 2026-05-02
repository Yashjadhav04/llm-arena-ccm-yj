# Metric Uncertainty (Session Bootstrap)

## Setup
- Bootstrap replications: 2000
- Resampling unit: evaluation session (with replacement)
- Compared models: `baseline`, `full_formatting`, `signal_interactions`

## Model Metric CIs
### Validation
- `log_loss`
  `baseline`: mean=0.645496, 95% CI [0.639665, 0.651512]
  `full_formatting`: mean=0.638052, 95% CI [0.631668, 0.644253]
  `signal_interactions`: mean=0.638318, 95% CI [0.631997, 0.644610]
- `accuracy`
  `baseline`: mean=0.626963, 95% CI [0.617267, 0.636151]
  `full_formatting`: mean=0.636710, 95% CI [0.627247, 0.646151]
  `signal_interactions`: mean=0.641481, 95% CI [0.631879, 0.651199]
- `brier_score`
  `baseline`: mean=0.226905, 95% CI [0.224225, 0.229662]
  `full_formatting`: mean=0.223183, 95% CI [0.220293, 0.225994]
  `signal_interactions`: mean=0.223238, 95% CI [0.220405, 0.226041]

### Test
- `log_loss`
  `baseline`: mean=0.646442, 95% CI [0.640629, 0.652471]
  `full_formatting`: mean=0.635599, 95% CI [0.629292, 0.641901]
  `signal_interactions`: mean=0.638740, 95% CI [0.632222, 0.645049]
- `accuracy`
  `baseline`: mean=0.627493, 95% CI [0.617892, 0.636698]
  `full_formatting`: mean=0.638334, 95% CI [0.628907, 0.647774]
  `signal_interactions`: mean=0.641865, 95% CI [0.632294, 0.650711]
- `brier_score`
  `baseline`: mean=0.227315, 95% CI [0.224618, 0.230088]
  `full_formatting`: mean=0.222348, 95% CI [0.219467, 0.225222]
  `signal_interactions`: mean=0.223204, 95% CI [0.220371, 0.226059]

## Delta CIs
- Delta definition: `model_a - model_b`
### Validation
- `log_loss`
  `full_formatting_minus_baseline`: mean=-0.007444, 95% CI [-0.010153, -0.004472]
  `signal_interactions_minus_full_formatting`: mean=0.000266, 95% CI [-0.001925, 0.002513]
- `accuracy`
  `full_formatting_minus_baseline`: mean=0.009747, 95% CI [0.003868, 0.015841]
  `signal_interactions_minus_full_formatting`: mean=0.004772, 95% CI [-0.000820, 0.010058]
- `brier_score`
  `full_formatting_minus_baseline`: mean=-0.003722, 95% CI [-0.004797, -0.002613]
  `signal_interactions_minus_full_formatting`: mean=0.000055, 95% CI [-0.000835, 0.000927]

### Test
- `log_loss`
  `full_formatting_minus_baseline`: mean=-0.010843, 95% CI [-0.013311, -0.008314]
  `signal_interactions_minus_full_formatting`: mean=0.003142, 95% CI [0.001055, 0.005348]
- `accuracy`
  `full_formatting_minus_baseline`: mean=0.010841, 95% CI [0.005010, 0.016835]
  `signal_interactions_minus_full_formatting`: mean=0.003531, 95% CI [-0.001519, 0.008604]
- `brier_score`
  `full_formatting_minus_baseline`: mean=-0.004966, 95% CI [-0.006041, -0.003895]
  `signal_interactions_minus_full_formatting`: mean=0.000856, 95% CI [-0.000032, 0.001755]

## Notes
- For `log_loss` and `brier_score`, lower is better.
- For `accuracy`, higher is better.
- If a delta CI includes 0, the difference is not clearly separable at this resolution.
