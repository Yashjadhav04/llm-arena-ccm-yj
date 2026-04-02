# Full Dataset Exploratory Data Analysis

## Dataset Overview
- Data source: lmarena-ai/arena-human-preference-140k [train, cache arrow]
- Rows: 135,634
- Unique vote ids: 135,634
- Duplicate vote ids: 0
- Unique models across both sides: 53
- Unique evaluation sessions: 115,372
- Binary votes (`model_a` or `model_b` wins): 98,348
- Non-binary votes (`tie` or `both_bad`): 37,286
- Time range: 2025-04-17 00:20:51.563409 to 2025-07-24 23:59:41.255781

## Feature Cardinality
| feature | dtype | non_null | missing | unique_non_null |
| --- | --- | --- | --- | --- |
| abs_length_diff_tokens | int64 | 135634 | 0 | 5580 |
| assistant_a_tokens | int64 | 135634 | 0 | 7100 |
| assistant_b_tokens | int64 | 135634 | 0 | 7084 |
| context_a_tokens | int64 | 135634 | 0 | 7649 |
| context_b_tokens | int64 | 135634 | 0 | 7682 |
| creative_writing | bool | 135634 | 0 | 2 |
| criteria_complexity | bool | 135634 | 0 | 2 |
| criteria_creativity | bool | 135634 | 0 | 2 |
| criteria_domain_knowledge | bool | 135634 | 0 | 2 |
| criteria_problem_solving | bool | 135634 | 0 | 2 |
| criteria_real_world | bool | 135634 | 0 | 2 |
| criteria_specificity | bool | 135634 | 0 | 2 |
| criteria_technical_accuracy | bool | 135634 | 0 | 2 |
| evaluation_order | int64 | 135634 | 0 | 28 |
| evaluation_session_id | object | 135634 | 0 | 115372 |
| id | object | 135634 | 0 | 135634 |
| instruction_following | bool | 135634 | 0 | 2 |
| is_binary_vote | bool | 135634 | 0 | 2 |
| is_code | bool | 135634 | 0 | 2 |
| language | object | 135634 | 0 | 126 |

The full cardinality table is saved to `results/eda_feature_cardinality.csv`.

## Outcome Distribution
| value | count | share |
| --- | --- | --- |
| model_b | 49785 | 0.3671 |
| model_a | 48563 | 0.3580 |
| tie | 21532 | 0.1588 |
| both_bad | 15754 | 0.1162 |

## Task Bucket Distribution
| value | count | share |
| --- | --- | --- |
| mixed | 70102 | 0.5168 |
| factual_reasoning | 52137 | 0.3844 |
| other | 9726 | 0.0717 |
| creative | 3669 | 0.0271 |

## Language Distribution
| value | count | share |
| --- | --- | --- |
| en | 71175 | 0.5334 |
| pl | 13813 | 0.1035 |
| und | 11835 | 0.0887 |
| ru | 9263 | 0.0694 |
| zh | 6501 | 0.0487 |
| de | 4311 | 0.0323 |
| ko | 2486 | 0.0186 |
| ja | 2444 | 0.0183 |
| fr | 2128 | 0.0159 |
| pt | 1684 | 0.0126 |
| es | 1672 | 0.0125 |
| fa | 1599 | 0.0120 |
| it | 925 | 0.0069 |
| zh-Hant | 845 | 0.0063 |
| vi | 660 | 0.0049 |

## Boolean Feature Prevalence
| feature | true_count | true_share | false_count | missing |
| --- | --- | --- | --- | --- |
| criteria_domain_knowledge | 109936 | 0.8105 | 25698 | 0 |
| is_binary_vote | 98348 | 0.7251 | 37286 | 0 |
| criteria_problem_solving | 92247 | 0.6801 | 43387 | 0 |
| criteria_technical_accuracy | 82881 | 0.6111 | 52753 | 0 |
| criteria_specificity | 80315 | 0.5921 | 55319 | 0 |
| criteria_real_world | 77642 | 0.5724 | 57992 | 0 |
| criteria_creativity | 71495 | 0.5271 | 64139 | 0 |
| criteria_complexity | 66438 | 0.4898 | 69196 | 0 |
| is_code | 39363 | 0.2902 | 96271 | 0 |
| instruction_following | 24666 | 0.1819 | 110968 | 0 |
| creative_writing | 11587 | 0.0854 | 124047 | 0 |
| math | 10892 | 0.0803 | 124742 | 0 |

## Numeric Feature Summary
| feature | non_null | missing | mean | std | min | p01 | p05 | p25 | median | p75 | p95 | p99 | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| evaluation_order | 135634 | 0 | 1.4107 | 1.2199 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.0000 | 7.0000 | 28.0000 |
| assistant_a_tokens | 135634 | 0 | 1149.7866 | 1716.1548 | 0.0000 | 13.0000 | 59.0000 | 322.0000 | 748.0000 | 1414.0000 | 3365.0000 | 7164.0100 | 88300.0000 |
| assistant_b_tokens | 135634 | 0 | 1149.1383 | 1734.3277 | 0.0000 | 12.0000 | 58.0000 | 318.0000 | 746.0000 | 1417.0000 | 3374.3500 | 7100.6900 | 137128.0000 |
| context_a_tokens | 135634 | 0 | 624.2761 | 2124.4854 | 1.0000 | 3.0000 | 6.0000 | 17.0000 | 45.0000 | 249.0000 | 3156.3500 | 9321.3500 | 116595.0000 |
| context_b_tokens | 135634 | 0 | 628.2380 | 2113.3002 | 1.0000 | 3.0000 | 6.0000 | 17.0000 | 45.0000 | 249.0000 | 3187.0000 | 9406.0900 | 79376.0000 |
| turns | 135634 | 0 | 1.2611 | 0.9969 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.0000 | 5.0000 | 64.0000 |
| winner_binary | 98348 | 37286 | 0.4938 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| length_diff_tokens | 135634 | 0 | 0.6484 | 1532.9115 | -69532.0000 | -3560.6700 | -1566.0000 | -365.0000 | 0.0000 | 368.0000 | 1550.0000 | 3582.3400 | 86773.0000 |
| abs_length_diff_tokens | 135634 | 0 | 696.0945 | 1365.7478 | 0.0000 | 2.0000 | 14.0000 | 122.0000 | 367.0000 | 833.0000 | 2262.3500 | 5079.3400 | 86773.0000 |
| log_length_ratio | 135634 | 0 | 0.0044 | 0.9629 | -8.3763 | -2.3781 | -1.5317 | -0.5390 | 0.0000 | 0.5496 | 1.5407 | 2.4260 | 8.8519 |
| length_diff_z | 135634 | 0 | 0.0000 | 1.0000 | -45.3599 | -2.3232 | -1.0220 | -0.2385 | -0.0004 | 0.2396 | 1.0107 | 2.3365 | 56.6062 |

## Session Structure
| sessions | mean_rows_per_session | median_rows_per_session | p95_rows_per_session | max_rows_per_session |
| --- | --- | --- | --- | --- |
| 115372 | 1.1756 | 1.0000 | 2.0000 | 25 |

### Session Size Distribution
| rows_in_session | session_count | share_of_sessions |
| --- | --- | --- |
| 1 | 102166 | 0.8855 |
| 2 | 9404 | 0.0815 |
| 3 | 2272 | 0.0197 |
| 4 | 790 | 0.0068 |
| 5 | 365 | 0.0032 |
| 6 | 163 | 0.0014 |
| 7 | 91 | 0.0008 |
| 8 | 35 | 0.0003 |
| 9 | 31 | 0.0003 |
| 10 | 17 | 0.0001 |
| 11 | 9 | 0.0001 |
| 12 | 10 | 0.0001 |
| 13 | 5 | 0.0000 |
| 14 | 4 | 0.0000 |
| 15 | 3 | 0.0000 |

## Evaluation Order Distribution
| value | count | share |
| --- | --- | --- |
| 1 | 108315 | 0.7987 |
| 2 | 15972 | 0.1178 |
| 3 | 5350 | 0.0395 |
| 4 | 2453 | 0.0181 |
| 5 | 1310 | 0.0097 |
| 6 | 744 | 0.0055 |
| 7 | 495 | 0.0037 |
| 8 | 304 | 0.0022 |
| 9 | 191 | 0.0014 |
| 10 | 138 | 0.0010 |
| 11 | 78 | 0.0006 |
| 12 | 63 | 0.0005 |
| 13 | 50 | 0.0004 |
| 14 | 39 | 0.0003 |
| 15 | 30 | 0.0002 |

## Response Length Buckets
### Assistant A Tokens
| bucket | count | share |
| --- | --- | --- |
| 1025+ | 50777 | 0.3744 |
| 513-1024 | 34387 | 0.2535 |
| 257-512 | 23571 | 0.1738 |
| 129-256 | 13038 | 0.0961 |
| <=64 | 7310 | 0.0539 |
| 65-128 | 6551 | 0.0483 |

### Assistant B Tokens
| bucket | count | share |
| --- | --- | --- |
| 1025+ | 50934 | 0.3755 |
| 513-1024 | 33991 | 0.2506 |
| 257-512 | 23496 | 0.1732 |
| 129-256 | 13161 | 0.0970 |
| <=64 | 7446 | 0.0549 |
| 65-128 | 6606 | 0.0487 |

## Model Frequency
| model | appearances | appearance_share |
| --- | --- | --- |
| claude-opus-4-20250514 | 10092 | 0.0372 |
| gemini-2.5-flash | 9668 | 0.0356 |
| gemini-2.5-pro | 9219 | 0.0340 |
| mistral-medium-2505 | 9135 | 0.0337 |
| qwen3-235b-a22b-no-thinking | 9076 | 0.0335 |
| o3-2025-04-16 | 8529 | 0.0314 |
| claude-sonnet-4-20250514 | 8295 | 0.0306 |
| chatgpt-4o-latest-20250326 | 7650 | 0.0282 |
| gemma-3-27b-it | 7326 | 0.0270 |
| claude-3-7-sonnet-20250219-thinking-32k | 7149 | 0.0264 |
| claude-3-7-sonnet-20250219 | 6852 | 0.0253 |
| command-a-03-2025 | 6838 | 0.0252 |
| claude-3-5-sonnet-20241022 | 6817 | 0.0251 |
| o3-mini | 6624 | 0.0244 |
| deepseek-r1-0528 | 6554 | 0.0242 |
| gpt-4.1-2025-04-14 | 6540 | 0.0241 |
| o4-mini-2025-04-16 | 6477 | 0.0239 |
| claude-3-5-haiku-20241022 | 6466 | 0.0238 |
| amazon.nova-pro-v1:0 | 6383 | 0.0235 |
| claude-opus-4-20250514-thinking-16k | 6157 | 0.0227 |

## Winner Distribution By Task Bucket
| task_bucket | both_bad | model_a | model_b | tie |
| --- | --- | --- | --- | --- |
| creative | 0.1398 | 0.3617 | 0.3734 | 0.1251 |
| factual_reasoning | 0.1200 | 0.3535 | 0.3606 | 0.1659 |
| mixed | 0.1118 | 0.3591 | 0.3738 | 0.1553 |
| other | 0.1182 | 0.3733 | 0.3506 | 0.1578 |

## Winner Distribution By `is_code`
| is_code | both_bad | model_a | model_b | tie |
| --- | --- | --- | --- | --- |
| False | 0.1149 | 0.3596 | 0.3682 | 0.1574 |
| True | 0.1193 | 0.3543 | 0.3643 | 0.1621 |

## Length And Outcome
### Length Difference By Winner
| winner | count | mean | median |
| --- | --- | --- | --- |
| both_bad | 15754 | 4.8272 | 0.0000 |
| model_a | 48563 | 261.7645 | 114.0000 |
| model_b | 49785 | -256.3594 | -116.0000 |
| tie | 21532 | 2.9108 | 0.0000 |

### Binary Winner Share Given Which Side Is Longer
| a_longer | model_a | model_b |
| --- | --- | --- |
| A longer | 0.6122 | 0.3878 |
| B longer | 0.3759 | 0.6241 |
| same length | 0.5418 | 0.4582 |

## Initial EDA Takeaways
- The dataset includes both binary preference votes and a substantial number of `tie` / `both_bad` outcomes, so multinomial handling will matter later.
- The task buckets are not balanced, with `mixed` and `factual_reasoning` dominating the first-pass split.
- Language is diverse enough that an English-only sensitivity analysis is worth doing before interpreting global coefficients.
- Response-length variables have wide tails, so robust scaling or log transforms are a good idea for modeling.
- Longer responses are descriptively associated with winning pairwise votes, which supports testing verbosity bias directly in the first cognitive model.
- Session and evaluation-order summaries can help decide whether to include session-level effects or clustered standard errors.
