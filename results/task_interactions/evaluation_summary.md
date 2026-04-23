# Task Interaction Evaluation

## Global Held-out Metrics
### Validation
- `baseline`: log_loss=0.6456, accuracy=0.6268, brier_score=0.2270
- `proxy_augmented`: log_loss=0.6399, accuracy=0.6377, brier_score=0.2241
- `formatting_augmented`: log_loss=0.6399, accuracy=0.6377, brier_score=0.2241
- `task_interactions`: log_loss=0.6400, accuracy=0.6376, brier_score=0.2240

### Test
- `baseline`: log_loss=0.6465, accuracy=0.6274, brier_score=0.2274
- `proxy_augmented`: log_loss=0.6415, accuracy=0.6362, brier_score=0.2245
- `formatting_augmented`: log_loss=0.6415, accuracy=0.6362, brier_score=0.2245
- `task_interactions`: log_loss=0.6416, accuracy=0.6374, brier_score=0.2246

## Held-out Delta: Task Interactions Minus Formatting-Augmented
- `validation` log_loss delta: 0.0001
- `validation` accuracy delta: -0.0001
- `test` log_loss delta: 0.0001
- `test` accuracy delta: 0.0012

## Per-task Held-out Log Loss
### Validation
- `mixed`: baseline=0.6353, proxy=0.6290, formatting=0.6290, task_interactions=0.6291
- `factual_reasoning`: baseline=0.6503, proxy=0.6462, formatting=0.6462, task_interactions=0.6459
- `creative`: baseline=0.6680, proxy=0.6600, formatting=0.6600, task_interactions=0.6592
- `other`: baseline=0.6868, proxy=0.6777, formatting=0.6777, task_interactions=0.6796

### Test
- `mixed`: baseline=0.6403, proxy=0.6342, formatting=0.6342, task_interactions=0.6342
- `factual_reasoning`: baseline=0.6487, proxy=0.6449, formatting=0.6449, task_interactions=0.6448
- `creative`: baseline=0.6684, proxy=0.6606, formatting=0.6606, task_interactions=0.6612
- `other`: baseline=0.6709, proxy=0.6678, formatting=0.6678, task_interactions=0.6688

## Split Task Counts
- `train` `mixed`: n=41,109, share=0.5225
- `train` `factual_reasoning`: n=29,851, share=0.3794
- `train` `other`: n=5,575, share=0.0709
- `train` `creative`: n=2,143, share=0.0272
- `validation` `mixed`: n=5,130, share=0.5216
- `validation` `factual_reasoning`: n=3,700, share=0.3762
- `validation` `other`: n=721, share=0.0733
- `validation` `creative`: n=284, share=0.0289
- `test` `mixed`: n=5,142, share=0.5228
- `test` `factual_reasoning`: n=3,678, share=0.3740
- `test` `other`: n=745, share=0.0757
- `test` `creative`: n=270, share=0.0275

## Task-specific Implied Slopes (Task Interaction Model)
- `mixed`: length_effect=0.2742, formatting_effect=0.0000
- `factual_reasoning`: length_effect=0.3807, formatting_effect=0.0000
- `creative`: length_effect=0.7373, formatting_effect=0.0000
- `other`: length_effect=0.1073, formatting_effect=0.0000

## Notes
- `mixed` is the reference task bucket for interactions.
- Per-task metrics are held-out only (`validation`, `test`).
