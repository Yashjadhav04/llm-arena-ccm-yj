# Creative/Factual Signal Interaction Evaluation

## Global Held-out Metrics
### Validation
- `baseline`: log_loss=0.6456, accuracy=0.6268, brier_score=0.2270
- `proxy_augmented`: log_loss=0.6399, accuracy=0.6377, brier_score=0.2241
- `formatting_augmented`: log_loss=0.6383, accuracy=0.6402, brier_score=0.2233
- `full_formatting`: log_loss=0.6382, accuracy=0.6365, brier_score=0.2233
- `full_formatting_plus_signals`: log_loss=0.6381, accuracy=0.6368, brier_score=0.2232
- `signal_main_effects`: log_loss=0.6381, accuracy=0.6406, brier_score=0.2232
- `signal_interactions`: log_loss=0.6385, accuracy=0.6413, brier_score=0.2233

### Test
- `baseline`: log_loss=0.6465, accuracy=0.6274, brier_score=0.2274
- `proxy_augmented`: log_loss=0.6415, accuracy=0.6362, brier_score=0.2245
- `formatting_augmented`: log_loss=0.6388, accuracy=0.6409, brier_score=0.2232
- `full_formatting`: log_loss=0.6357, accuracy=0.6382, brier_score=0.2224
- `full_formatting_plus_signals`: log_loss=0.6359, accuracy=0.6393, brier_score=0.2225
- `signal_main_effects`: log_loss=0.6389, accuracy=0.6411, brier_score=0.2233
- `signal_interactions`: log_loss=0.6388, accuracy=0.6417, brier_score=0.2232

## Held-out Delta: Signal Interactions Minus Full Formatting
- `validation` log_loss delta: 0.0003
- `validation` accuracy delta: 0.0048
- `test` log_loss delta: 0.0031
- `test` accuracy delta: 0.0035

## Per-task Held-out Log Loss
### Validation
- `mixed`: baseline=0.6353, proxy_augmented=0.6290, formatting_augmented=0.6281, full_formatting=0.6284, full_formatting_plus_signals=0.6283, signal_main_effects=0.6280, signal_interactions=0.6282
- `factual_reasoning`: baseline=0.6503, proxy_augmented=0.6462, formatting_augmented=0.6444, full_formatting=0.6438, full_formatting_plus_signals=0.6438, signal_main_effects=0.6444, signal_interactions=0.6441
- `creative`: baseline=0.6680, proxy_augmented=0.6600, formatting_augmented=0.6632, full_formatting=0.6594, full_formatting_plus_signals=0.6589, signal_main_effects=0.6633, signal_interactions=0.6624
- `other`: baseline=0.6868, proxy_augmented=0.6777, formatting_augmented=0.6693, full_formatting=0.6708, full_formatting_plus_signals=0.6696, signal_main_effects=0.6678, signal_interactions=0.6730

### Test
- `mixed`: baseline=0.6403, proxy_augmented=0.6342, formatting_augmented=0.6312, full_formatting=0.6272, full_formatting_plus_signals=0.6273, signal_main_effects=0.6312, signal_interactions=0.6312
- `factual_reasoning`: baseline=0.6487, proxy_augmented=0.6449, formatting_augmented=0.6423, full_formatting=0.6399, full_formatting_plus_signals=0.6399, signal_main_effects=0.6423, signal_interactions=0.6422
- `creative`: baseline=0.6684, proxy_augmented=0.6606, formatting_augmented=0.6583, full_formatting=0.6590, full_formatting_plus_signals=0.6590, signal_main_effects=0.6587, signal_interactions=0.6596
- `other`: baseline=0.6709, proxy_augmented=0.6678, formatting_augmented=0.6675, full_formatting=0.6649, full_formatting_plus_signals=0.6672, signal_main_effects=0.6688, signal_interactions=0.6676

## Per-signal-state Held-out Log Loss
### Validation
- `none`: baseline=0.6868, proxy_augmented=0.6777, formatting_augmented=0.6693, full_formatting=0.6708, full_formatting_plus_signals=0.6696, signal_main_effects=0.6678, signal_interactions=0.6730
- `creative_only`: baseline=0.6680, proxy_augmented=0.6600, formatting_augmented=0.6632, full_formatting=0.6594, full_formatting_plus_signals=0.6589, signal_main_effects=0.6633, signal_interactions=0.6624
- `factual_only`: baseline=0.6503, proxy_augmented=0.6462, formatting_augmented=0.6444, full_formatting=0.6438, full_formatting_plus_signals=0.6438, signal_main_effects=0.6444, signal_interactions=0.6441
- `both`: baseline=0.6353, proxy_augmented=0.6290, formatting_augmented=0.6281, full_formatting=0.6284, full_formatting_plus_signals=0.6283, signal_main_effects=0.6280, signal_interactions=0.6282

### Test
- `none`: baseline=0.6709, proxy_augmented=0.6678, formatting_augmented=0.6675, full_formatting=0.6649, full_formatting_plus_signals=0.6672, signal_main_effects=0.6688, signal_interactions=0.6676
- `creative_only`: baseline=0.6684, proxy_augmented=0.6606, formatting_augmented=0.6583, full_formatting=0.6590, full_formatting_plus_signals=0.6590, signal_main_effects=0.6587, signal_interactions=0.6596
- `factual_only`: baseline=0.6487, proxy_augmented=0.6449, formatting_augmented=0.6423, full_formatting=0.6399, full_formatting_plus_signals=0.6399, signal_main_effects=0.6423, signal_interactions=0.6422
- `both`: baseline=0.6403, proxy_augmented=0.6342, formatting_augmented=0.6312, full_formatting=0.6272, full_formatting_plus_signals=0.6273, signal_main_effects=0.6312, signal_interactions=0.6312

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

## Split Signal-State Counts
- `train` `both`: n=41,109, share=0.5225
- `train` `factual_only`: n=29,851, share=0.3794
- `train` `none`: n=5,575, share=0.0709
- `train` `creative_only`: n=2,143, share=0.0272
- `validation` `both`: n=5,130, share=0.5216
- `validation` `factual_only`: n=3,700, share=0.3762
- `validation` `none`: n=721, share=0.0733
- `validation` `creative_only`: n=284, share=0.0289
- `test` `both`: n=5,142, share=0.5228
- `test` `factual_only`: n=3,678, share=0.3740
- `test` `none`: n=745, share=0.0757
- `test` `creative_only`: n=270, share=0.0275

## Signal-state Implied Slopes (Signal Interaction Model)
- `none`: length_effect=0.0898, formatting_effect=0.1789
- `creative_only`: length_effect=0.6266, formatting_effect=0.2859
- `factual_only`: length_effect=0.3585, formatting_effect=0.3210
- `both`: length_effect=0.2697, formatting_effect=0.3103

## Notes
- `creative_signal` and `factual_signal` are non-exclusive and can both equal 1.
- `both_signal = creative_signal * factual_signal` captures overlap effects.
- Per-task metrics are held-out only (`validation`, `test`).