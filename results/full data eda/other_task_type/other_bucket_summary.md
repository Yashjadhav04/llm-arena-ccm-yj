# Other Bucket Exploration

## Data Scope
- Source for split data: `/Users/maingoclanvy/ccm/results/train_val_test_evaluation`
- Source for raw prompt extraction: `lmarena-ai/arena-human-preference-140k [train, cache arrow]`
- Total rows: 98,348
- `other` rows: 7,041 (7.16%)

## Subtype Breakdown Inside `other`
- `none`: n=6,878, share=97.68%
- `specificity`: n=69, share=0.98%
- `complexity+specificity`: n=45, share=0.64%
- `complexity+real_world`: n=14, share=0.20%
- `complexity`: n=13, share=0.18%
- `real_world`: n=13, share=0.18%
- `complexity+specificity+real_world`: n=6, share=0.09%
- `specificity+real_world`: n=3, share=0.04%

## Lexical Prompt Heuristic Inside `other`
- `unclear`: n=6,306, share=89.56%
- `factual_like`: n=680, share=9.66%
- `creative_like`: n=37, share=0.53%
- `creative_and_factual_like`: n=18, share=0.26%

## Top Prompt Tokens (Document Frequency)
- `hello`: 146
- `best`: 142
- `one`: 132
- `know`: 115
- `there`: 112
- `what's`: 112
- `does`: 110
- `lmarena`: 101
- `today`: 100
- `i'm`: 98
- `time`: 97
- `think`: 96
- `life`: 93
- `more`: 90
- `jest`: 85
- `want`: 83
- `czy`: 83
- `been`: 81
- `hey`: 81
- `day`: 78

## Recommendation
- Keep `creative` vs `factual_reasoning` as the main split; do not relabel `other` rows into those classes automatically.
- Use `criteria_complexity`, `criteria_specificity`, and `criteria_real_world` as orthogonal covariates (or interaction features) rather than forcing a new exclusive task label.
- If you want one additional task type, define a narrow `pragmatic_other` type from these criteria and test it as a sensitivity analysis.

