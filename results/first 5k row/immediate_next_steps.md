# Immediate Next Steps Report

## What This Run Covered
- Data source: lmarena-ai/arena-human-preference-140k [train[:5000], cache arrow]
- Raw rows loaded: 5,000 
- Binary rows used for the first models: 3,684
- Immediate next steps executed:
  - confirmed usable fields
  - set an initial task split
  - fit a baseline Bradley-Terry style model
  - fit a first bias model with position and length terms

## Confirmed First-Pass Fields
- Outcome: `model_a`, `model_b`, `winner`
- Session context: `evaluation_session_id`, `evaluation_order`, `language`, `timestamp`
- Length features: `conv_metadata.sum_assistant_a_tokens`, `conv_metadata.sum_assistant_b_tokens`
- Task split signals: `category_tag` subfields plus `is_code`

## Recommended First Task Split
- `creative`: creative writing or creativity signal
- `factual_reasoning`: math, instruction following, code, problem solving, domain knowledge, or technical accuracy signal
- `mixed`: both signal types
- `other`: neither signal type

### Task Bucket Counts
- `mixed`: 2,597
- `factual_reasoning`: 1,892
- `other`: 380
- `creative`: 131

### Winner Label Counts
- `model_b`: 1,861
- `model_a`: 1,823
- `tie`: 787
- `both_bad`: 529

### Top Languages
- `en`: 2,593
- `pl`: 500
- `und`: 462
- `ru`: 334
- `zh`: 251
- `de`: 173
- `ja`: 89
- `ko`: 85
- `fr`: 63
- `fa`: 63

## Baseline Model
- Rows: 3,684, models: 52, log loss: 0.6367, accuracy: 0.6409
- Top models by baseline score:
  - `gemini-2.0-flash-thinking-exp-01-21`: 1.3510
  - `gemini-2.5-pro-preview-03-25`: 1.1622
  - `chatgpt-4o-latest-20250326`: 0.8634
  - `gemini-2.5-pro`: 0.7951
  - `gemini-2.5-pro-preview-05-06`: 0.6847
  - `deepseek-r1-0528`: 0.6408
  - `gemini-2.5-flash`: 0.6206
  - `llama-4-maverick-03-26-experimental`: 0.5486
  - `o3-2025-04-16`: 0.5163
  - `grok-3-preview-02-24`: 0.4622

## First Bias Model
- Rows: 3,684, models: 52, log loss: 0.6328, accuracy: 0.6455
- Side-A position coefficient: -0.0401
- Standardized length-difference coefficient: 0.2961
- Top models by bias-adjusted score:
  - `gemini-2.0-flash-thinking-exp-01-21`: 1.3516
  - `gemini-2.5-pro-preview-03-25`: 1.0466
  - `chatgpt-4o-latest-20250326`: 0.8677
  - `gemini-2.5-pro`: 0.6877
  - `deepseek-r1-0528`: 0.5769
  - `gemini-2.5-pro-preview-05-06`: 0.5423
  - `o3-2025-04-16`: 0.5010
  - `o4-mini-2025-04-16`: 0.4847
  - `gemini-2.5-flash`: 0.4834
  - `llama-4-maverick-03-26-experimental`: 0.4235

## Task-Specific Bias Check
- `creative` (small-sample): n=93, status=ok, side_a_bias=0.7826, length_diff_z=2.2958
- `factual_reasoning`: n=1,358, status=ok, side_a_bias=0.0310, length_diff_z=0.8026

## Interpretation
- The baseline model gives the first leaderboard estimate using model identity alone.
- The bias model estimates whether side-A position and response length help explain votes beyond model identity.
- The task split is ready for the next paper pass comparing creative versus factual/reasoning prompts.
- These estimates are provisional because this run used the first 5,000 rows rather than the full 135,634-row dataset.

## Recommended Next Actions
- Extend the bias model with task interactions such as `length_diff_z * creative`.
- Decide whether to keep only English rows or explicitly compare languages.
- Add tie and `both_bad` handling after the binary comparison results are stable.
- Start drafting the methods section around the baseline-versus-bias comparison.
