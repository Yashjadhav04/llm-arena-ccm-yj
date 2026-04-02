# Arena Dataset Notes

## Confirmed Dataset
- Dataset: `lmarena-ai/arena-human-preference-140k`
- Split: `train`
- Size: 135,634 rows

## Confirmed Top-Level Fields
- `id`
- `model_a`
- `model_b`
- `winner`
- `evaluation_session_id`
- `evaluation_order`
- `conversation_a`
- `conversation_b`
- `full_conversation`
- `conv_metadata`
- `category_tag`
- `language`
- `is_code`
- `timestamp`

## Most Useful Fields For The First Analysis Pass
### Pairwise preference outcome
- `model_a`
- `model_b`
- `winner`

### Session and context controls
- `evaluation_session_id`
- `evaluation_order`
- `language`
- `timestamp`

### Length and style features
- `conv_metadata.sum_assistant_a_tokens`
- `conv_metadata.sum_assistant_b_tokens`
- `conv_metadata.context_a_tokens`
- `conv_metadata.context_b_tokens`
- `conv_metadata.turns`
- header, list, and bold count summaries inside `conv_metadata`

### Task split signals
- `category_tag.creative_writing_v0.1.creative_writing`
- `category_tag.criteria_v0.1.creativity`
- `category_tag.criteria_v0.1.problem_solving`
- `category_tag.criteria_v0.1.domain_knowledge`
- `category_tag.criteria_v0.1.technical_accuracy`
- `category_tag.if_v0.1.if`
- `category_tag.math_v0.1.math`
- `is_code`

## Recommended First Task Split
For the first project pass, use a simple and defensible split:

- `creative`
  - `creative_writing` is true, or the `creativity` criterion is true
- `factual_reasoning`
  - `math`, `if`, `is_code`, `problem_solving`, `domain_knowledge`, or `technical_accuracy` is true
- `mixed`
  - both creative and factual/reasoning signals are true
- `other`
  - neither set of signals is present

For the first paper analysis, compare `creative` against `factual_reasoning`, and treat `mixed` and `other` as secondary buckets unless they are large and interpretable enough to analyze directly.

## Initial Modeling Recommendation
Use binary rows only for the first model pass:

- keep `winner == model_a` or `winner == model_b`
- hold out `tie` and `both_bad` for later extensions

Fit two models:

1. Baseline Bradley-Terry style model
   - model identity only

2. First cognitive-bias model
   - model identity
   - side-A intercept as a first-pass position bias estimate
   - standardized assistant token length difference as the first verbosity bias estimate

## Why This Split Works
- It stays close to the professor feedback about testing interpretable features like length and position.
- It stays close to the TA feedback about contrasting task types.
- It avoids overfitting the first pass with too many categories before the data distribution is understood.

