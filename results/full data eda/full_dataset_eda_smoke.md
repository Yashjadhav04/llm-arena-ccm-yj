# Full Dataset Exploratory Data Analysis

## Dataset Overview
- Data source: lmarena-ai/arena-human-preference-140k [train[:1000], cache arrow]
- Rows: 1,000
- Unique vote ids: 1,000
- Duplicate vote ids: 0
- Unique models across both sides: 52
- Unique evaluation sessions: 1,000
- Binary votes (`model_a` or `model_b` wins): 736
- Non-binary votes (`tie` or `both_bad`): 264
- Time range: 2025-04-17 17:59:50.819568 to 2025-07-24 20:37:20.706392

## Feature Cardinality
| feature | dtype | non_null | missing | unique_non_null |
| --- | --- | --- | --- | --- |
| abs_length_diff_tokens | int64 | 1000 | 0 | 688 |
| assistant_a_tokens | int64 | 1000 | 0 | 815 |
| assistant_b_tokens | int64 | 1000 | 0 | 804 |
| context_a_tokens | int64 | 1000 | 0 | 393 |
| context_b_tokens | int64 | 1000 | 0 | 398 |
| creative_writing | bool | 1000 | 0 | 2 |
| criteria_complexity | bool | 1000 | 0 | 2 |
| criteria_creativity | bool | 1000 | 0 | 2 |
| criteria_domain_knowledge | bool | 1000 | 0 | 2 |
| criteria_problem_solving | bool | 1000 | 0 | 2 |
| criteria_real_world | bool | 1000 | 0 | 2 |
| criteria_specificity | bool | 1000 | 0 | 2 |
| criteria_technical_accuracy | bool | 1000 | 0 | 2 |
| evaluation_order | int64 | 1000 | 0 | 11 |
| evaluation_session_id | object | 1000 | 0 | 1000 |
| id | object | 1000 | 0 | 1000 |
| instruction_following | bool | 1000 | 0 | 2 |
| is_binary_vote | bool | 1000 | 0 | 2 |
| is_code | bool | 1000 | 0 | 2 |
| language | object | 1000 | 0 | 34 |

The full cardinality table is saved to `results/eda_feature_cardinality.csv`.

## Outcome Distribution
| value | count | share |
| --- | --- | --- |
| model_a | 380 | 0.3800 |
| model_b | 356 | 0.3560 |
| tie | 157 | 0.1570 |
| both_bad | 107 | 0.1070 |

## Task Bucket Distribution
| value | count | share |
| --- | --- | --- |
| mixed | 506 | 0.5060 |
| factual_reasoning | 396 | 0.3960 |
| other | 70 | 0.0700 |
| creative | 28 | 0.0280 |

## Language Distribution
| value | count | share |
| --- | --- | --- |
| en | 529 | 0.5409 |
| und | 95 | 0.0971 |
| pl | 91 | 0.0930 |
| ru | 63 | 0.0644 |
| de | 44 | 0.0450 |
| zh | 40 | 0.0409 |
| ja | 20 | 0.0204 |
| fr | 15 | 0.0153 |
| ko | 13 | 0.0133 |
| pt | 13 | 0.0133 |
| es | 12 | 0.0123 |
| it | 8 | 0.0082 |
| tr | 7 | 0.0072 |
| fa | 6 | 0.0061 |
| zh-Hant | 5 | 0.0051 |

## Boolean Feature Prevalence
| feature | true_count | true_share | false_count | missing |
| --- | --- | --- | --- | --- |
| criteria_domain_knowledge | 805 | 0.8050 | 195 | 0 |
| is_binary_vote | 736 | 0.7360 | 264 | 0 |
| criteria_problem_solving | 668 | 0.6680 | 332 | 0 |
| criteria_technical_accuracy | 606 | 0.6060 | 394 | 0 |
| criteria_specificity | 601 | 0.6010 | 399 | 0 |
| criteria_real_world | 567 | 0.5670 | 433 | 0 |
| criteria_creativity | 515 | 0.5150 | 485 | 0 |
| criteria_complexity | 469 | 0.4690 | 531 | 0 |
| is_code | 296 | 0.2960 | 704 | 0 |
| instruction_following | 175 | 0.1750 | 825 | 0 |
| creative_writing | 99 | 0.0990 | 901 | 0 |
| math | 69 | 0.0690 | 931 | 0 |

## Numeric Feature Summary
| feature | non_null | missing | mean | std | min | p01 | p05 | p25 | median | p75 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| evaluation_order | 1000 | 0 | 1.4110 | 1.1408 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.0000 | 7.0000 | 12.0000 |
| assistant_a_tokens | 1000 | 0 | 1205.1380 | 2254.0923 | 1.0000 | 11.9800 | 57.9500 | 313.5000 | 788.0000 | 1427.0000 | 3246.3500 | 7870.3600 | 51560.0000 |
| assistant_b_tokens | 1000 | 0 | 1073.8970 | 1340.0663 | 1.0000 | 9.9900 | 56.9500 | 296.7500 | 733.0000 | 1381.0000 | 3158.2500 | 6334.8700 | 19331.0000 |
| context_a_tokens | 1000 | 0 | 562.6720 | 1686.9570 | 1.0000 | 3.0000 | 6.0000 | 16.0000 | 40.0000 | 196.5000 | 2928.6000 | 7926.9000 | 17288.0000 |
| context_b_tokens | 1000 | 0 | 568.1090 | 1709.1944 | 1.0000 | 3.0000 | 6.0000 | 16.0000 | 40.0000 | 217.5000 | 3192.9000 | 8152.2900 | 20308.0000 |
| turns | 1000 | 0 | 1.2740 | 1.1385 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2.0000 | 5.0000 | 24.0000 |
| winner_binary | 736 | 264 | 0.5163 | 0.5001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| length_diff_tokens | 1000 | 0 | 131.2410 | 2033.4769 | -7883.0000 | -3109.1200 | -1318.3500 | -295.5000 | 18.0000 | 439.5000 | 1503.3000 | 3643.0100 | 50627.0000 |
| abs_length_diff_tokens | 1000 | 0 | 708.0310 | 1910.6177 | 0.0000 | 2.0000 | 16.9500 | 120.7500 | 368.5000 | 806.2500 | 2203.6500 | 5078.1400 | 50627.0000 |
| log_length_ratio | 1000 | 0 | 0.0576 | 0.9177 | -5.0081 | -2.1593 | -1.3896 | -0.4949 | 0.0642 | 0.5840 | 1.5428 | 2.1691 | 4.5346 |
| length_diff_z | 1000 | 0 | 0.0000 | 1.0000 | -3.9412 | -1.5935 | -0.7129 | -0.2099 | -0.0557 | 0.1516 | 0.6747 | 1.7270 | 24.8322 |

## Session Structure
| sessions | mean_rows_per_session | median_rows_per_session | p95_rows_per_session | max_rows_per_session |
| --- | --- | --- | --- | --- |
| 1000 | 1.0000 | 1.0000 | 1.0000 | 1 |

### Session Size Distribution
| rows_in_session | session_count | share_of_sessions |
| --- | --- | --- |
| 1 | 1000 | 1.0000 |

## Evaluation Order Distribution
| value | count | share |
| --- | --- | --- |
| 1 | 800 | 0.8000 |
| 2 | 112 | 0.1120 |
| 3 | 42 | 0.0420 |
| 5 | 16 | 0.0160 |
| 4 | 15 | 0.0150 |
| 9 | 5 | 0.0050 |
| 6 | 4 | 0.0040 |
| 7 | 3 | 0.0030 |
| 8 | 1 | 0.0010 |
| 12 | 1 | 0.0010 |
| 11 | 1 | 0.0010 |

## Response Length Buckets
### Assistant A Tokens
| bucket | count | share |
| --- | --- | --- |
| 1025+ | 388 | 0.3880 |
| 513-1024 | 236 | 0.2360 |
| 257-512 | 171 | 0.1710 |
| 129-256 | 98 | 0.0980 |
| <=64 | 54 | 0.0540 |
| 65-128 | 53 | 0.0530 |

### Assistant B Tokens
| bucket | count | share |
| --- | --- | --- |
| 1025+ | 354 | 0.3540 |
| 513-1024 | 248 | 0.2480 |
| 257-512 | 181 | 0.1810 |
| 129-256 | 107 | 0.1070 |
| <=64 | 56 | 0.0560 |
| 65-128 | 54 | 0.0540 |

## Model Frequency
| model | appearances | appearance_share |
| --- | --- | --- |
| gemini-2.5-flash | 81 | 0.0405 |
| claude-opus-4-20250514 | 76 | 0.0380 |
| claude-3-7-sonnet-20250219 | 73 | 0.0365 |
| qwen3-235b-a22b-no-thinking | 70 | 0.0350 |
| o3-2025-04-16 | 70 | 0.0350 |
| gemini-2.5-pro | 65 | 0.0325 |
| mistral-medium-2505 | 64 | 0.0320 |
| gemma-3-27b-it | 62 | 0.0310 |
| deepseek-r1-0528 | 59 | 0.0295 |
| claude-3-5-sonnet-20241022 | 55 | 0.0275 |
| amazon.nova-pro-v1:0 | 53 | 0.0265 |
| command-a-03-2025 | 53 | 0.0265 |
| o4-mini-2025-04-16 | 53 | 0.0265 |
| grok-3-mini-beta | 52 | 0.0260 |
| chatgpt-4o-latest-20250326 | 51 | 0.0255 |
| grok-3-preview-02-24 | 50 | 0.0250 |
| qwen3-30b-a3b | 48 | 0.0240 |
| gpt-4.1-2025-04-14 | 47 | 0.0235 |
| claude-3-7-sonnet-20250219-thinking-32k | 47 | 0.0235 |
| claude-sonnet-4-20250514 | 47 | 0.0235 |

## Winner Distribution By Task Bucket
| task_bucket | both_bad | model_a | model_b | tie |
| --- | --- | --- | --- | --- |
| creative | 0.1786 | 0.4643 | 0.2500 | 0.1071 |
| factual_reasoning | 0.1061 | 0.3813 | 0.3434 | 0.1692 |
| mixed | 0.1067 | 0.3656 | 0.3755 | 0.1522 |
| other | 0.0857 | 0.4429 | 0.3286 | 0.1429 |

## Winner Distribution By `is_code`
| is_code | both_bad | model_a | model_b | tie |
| --- | --- | --- | --- | --- |
| False | 0.1065 | 0.3821 | 0.3636 | 0.1477 |
| True | 0.1081 | 0.3750 | 0.3378 | 0.1791 |

## Initial EDA Takeaways
- The dataset includes both binary preference votes and a substantial number of `tie` / `both_bad` outcomes, so multinomial handling will matter later.
- The task buckets are not balanced, with `mixed` and `factual_reasoning` dominating the first-pass split.
- Language is diverse enough that an English-only sensitivity analysis is worth doing before interpreting global coefficients.
- Response-length variables have wide tails, so robust scaling or log transforms are a good idea for modeling.
- Session and evaluation-order summaries can help decide whether to include session-level effects or clustered standard errors.
