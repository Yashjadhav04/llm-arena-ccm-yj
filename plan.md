# Computational Cognitive Modelling Project Plan

## Working Title
Cognitive Bias in Arena-Style LLM Evaluation: Modeling Human Preferences Beyond Raw Votes

## Project Summary
This project asks whether human votes in Chatbot Arena-like evaluation actually measure latent model quality, or whether they are systematically shaped by cognitive biases in the way people judge responses. The core idea is to treat arena voting as a constructive preference process rather than as a direct readout of objective quality. Using the Arena human preference dataset, the project will model pairwise votes as a combination of true model quality and bias-related features such as response length, presentation order, and task framing. The final goal is to estimate a debiased ranking of models and test whether LLM-as-a-Judge agrees more with raw human votes or with debiased latent quality.

## Core Research Question
Do arena-style human preference votes reflect true model quality, or are they partly artifacts of cognitive biases such as verbosity preference, position bias, and task-dependent framing?

## Motivation
Human voting is often treated as the gold standard for LLM evaluation, but that assumption may be too strong. If arena judgments are influenced by systematic biases shared across voters, then aggregating more votes will not necessarily recover true quality. Instead, it may amplify distorted preferences. This makes arena evaluation a cognitive measurement problem, not just a ranking problem.

This framing fits well with preference construction theory from cognitive science, especially the idea that preferences are built in the moment and depend on context. It also matches the project feedback:

- The professor suggested focusing on features like length and position and testing whether they significantly predict preference.
- The TA suggested narrowing by task type and comparing different classes of prompts.
- Both sets of feedback point toward an explainable model of human judgment rather than a purely predictive one.

## Main Objectives
1. Build a baseline model of arena preferences using pairwise comparison methods.
2. Add interpretable cognitive-bias features to explain why one response is chosen over another.
3. Measure whether bias effects differ across task types such as creative writing versus factual or reasoning prompts.
4. Construct a debiased leaderboard and compare it to the raw leaderboard.
5. Evaluate whether LLM-as-a-Judge tracks raw human preference or debiased latent quality.

## Proposed Hypotheses
1. Response length will significantly affect human preference even after controlling for model identity.
2. Presentation order or position will have a measurable effect on which response is preferred.
3. Bias effects will vary by task type, with stronger stylistic bias in open-ended or creative prompts than in factual prompts.
4. A model that includes cognitive bias features will fit human voting data better than a baseline model without them.
5. Removing estimated bias terms will change at least some model rankings in a meaningful way.
6. LLM-as-a-Judge will partially replicate the same biases found in human voting rather than perfectly recovering debiased quality.

## Dataset and Scope
### Primary dataset
- `lmarena-ai/arena-human-preference-140k`

### Unit of analysis
- One pairwise comparison between response A and response B for a given prompt.

### Initial scope
- English or primarily text-based arena data
- Pairwise human preference outcomes
- Core bias features: length, position, framing/task type, and interpretable lexical or stylistic cues

### Optional extensions if time allows
- Multilingual analysis
- More detailed annotator-level modeling
- Small controlled intervention study or re-rating experiment

## Planned Comparison Structure
The project needs a clear answer to "compare to what?" The comparison structure will be:

1. Baseline leaderboard:
Standard pairwise aggregation without explicit bias parameters.

2. Cognitive-bias model:
Pairwise model with interpretable covariates for human judgment biases.

3. Debiased leaderboard:
Counterfactual ranking produced by removing the estimated influence of bias terms.

4. LLM-as-a-Judge comparison:
Measure whether LLM judges align more with raw human votes or with the debiased estimate of model quality.

## Methodological Plan
### Phase 1: Literature framing and theory
Ground the project in preference construction and crowd judgment research:

- Slovic on constructed preferences
- Michael Lee's work on cognitive models, debiasing, ranking, and correlated crowd bias
- Recent work on arena biases, especially verbosity and position effects

This phase will justify why arena evaluation should be treated as a cognitive modeling problem.

### Phase 2: Data understanding and preprocessing
Start by understanding what kinds of prompts and responses appear in the arena dataset.

Tasks:
- Inspect available fields in the dataset
- Clean pairwise outcome labels
- Identify what metadata can be used for prompt type or evaluation context
- Create a working task taxonomy, likely starting with:
  - Creative/open-ended prompts
  - Factual/reasoning prompts
  - Possibly code or instruction-following prompts if enough data exist

This stage matters because the TA feedback suggests that bias structure may differ across task classes.

### Phase 3: Feature construction
Construct interpretable observable features that may reflect latent evaluation heuristics or serve as control variables. The key idea is to avoid treating response length, side position, or task type as cognitive variables themselves. Instead, they will be modeled as surface cues or contextual moderators that may correlate with hidden judgment tendencies.

Core observable cues and moderators:
- Relative response length proxies
  - token count
  - character count
  - length ratio or length difference
  - interpreted as possible correlates of verbosity, completeness, effort, or informativeness heuristics
- Response position and presentation cues
  - side A versus side B indicator
  - evaluation order when available
  - interpreted as possible correlates of primacy effects, side preference, or attention asymmetry
- Task-context moderators
  - prompt type or task category
  - interactions such as length by task type or position by task type
  - used to test whether people rely on different evaluation heuristics in creative, factual, reasoning, or code tasks
- Style and explainability cues
  - n-grams
  - conjunctions or feature combinations
  - sequential information if conversation structure is available
  - interpreted as proxies for perceived clarity, confidence, structure, or informativeness

Control features:
- Readability or coherence proxies
- Safety or obvious failure indicators
- Topic or prompt-difficulty proxies where possible

The goal is not just prediction. The features should be interpretable enough to support claims of the form: votes change in systematic ways when an observable cue changes, which is consistent with a latent evaluation heuristic. Model identity terms will capture latent quality, while cue terms will estimate how votes shift with measurable surface characteristics.

### Phase 4: Baseline model
Fit a baseline pairwise preference model, such as Bradley-Terry or logistic pairwise regression using model identity alone.

Purpose:
- Provide a standard leaderboard estimate
- Establish a comparison point for the cognitive model

### Phase 5: Cognitive bias model
Fit an extended model where the probability that response A is preferred over response B depends on both latent quality and bias features.

Conceptual form:

`P(A preferred over B) = f(quality gap + verbosity bias + position bias + framing/task effects + noise)`

A concrete version could be:

`logit(P(A wins)) = alpha_A - alpha_B + beta_length * length_diff + beta_position * order + beta_task * task_features + beta_interactions + error`

This is the main modeling contribution of the project.

### Phase 6: Task-specific analysis
Compare bias parameters across task categories.

Examples:
- Are people more likely to reward verbosity for creative writing than for factual QA?
- Does position bias become stronger when prompts are subjective?
- Do stylistic cues matter more when there is no clear objective answer?

This helps connect the computational model to cognitive theory rather than treating bias as a single global parameter.

### Phase 7: Debiased ranking analysis
Use the fitted model to estimate what rankings would look like if bias terms were set to zero or otherwise neutralized.

Outputs:
- Raw leaderboard
- Debiased leaderboard
- Ranking shifts for major models
- Correlation and top-k overlap between the two leaderboards

This is likely to be one of the clearest and most compelling result sections in the final paper.

### Phase 8: LLM-as-a-Judge comparison
Sample a subset of prompt-response pairs and have a stronger LLM act as judge.

Compare LLM judgments against:
- Raw human labels
- Debiased latent quality estimates

Key question:
- Does LLM-as-a-Judge reproduce human cognitive bias, or does it recover something closer to latent quality?

### Phase 9: Controlled experiment if feasible
If time allows, add a small intervention-based experiment to isolate specific bias effects more cleanly.

Possible manipulations:
- Same content with different response lengths
- Same two responses with randomized order
- Style changes such as confidence or formality while preserving content
- A sanity check using scrambled or nonsensical text to test whether superficial cues dominate judgment

This would strengthen causal interpretation, especially for the professor's suggestion to isolate one feature at a time.

## Expected Deliverables
1. A final project paper with introduction, related work, method, results, and discussion.
2. A cleaned and documented analysis workflow.
3. At least one baseline model and one cognitive-bias model.
4. Figures showing:
   - important bias coefficients
   - differences across task types
   - raw versus debiased leaderboards
   - LLM-as-a-Judge agreement patterns
5. A short discussion of limitations, especially self-selection bias and unobserved confounds.

## Evaluation Criteria
The project will be successful if it can show most of the following:

- Bias-related features significantly predict preference outcomes
- The cognitive model improves on the baseline model
- Bias effects are interpretable and theoretically motivated
- Ranking changes after debiasing are non-trivial
- The analysis provides a clear answer to whether arena votes should be treated as direct measurements of model quality

## Risks and Mitigations
### Risk 1: Too many possible features
Mitigation:
Keep the first model focused on a few theory-driven features, especially length, position, and task type.

### Risk 2: Weak task labels in the dataset
Mitigation:
Use a simple and defensible task split rather than over-engineering fine-grained categories.

### Risk 3: Difficulty separating quality from bias
Mitigation:
Use a baseline-versus-extended comparison and interpret the model as estimating bias conditional on observed model pairings, not as perfect ground truth recovery.

### Risk 4: Project scope grows too large
Mitigation:
Treat multilingual analysis and controlled experiments as optional extensions, not core requirements.

## Recommended Milestone Timeline
### Week 1
- Read the most relevant cognitive modeling and arena-bias papers
- Inspect the dataset structure
- Finalize research question and hypotheses

### Week 2
- Build preprocessing pipeline
- Define task categories
- Engineer first-pass features

### Week 3
- Fit baseline model
- Fit cognitive-bias model
- Interpret core coefficients

### Week 4
- Generate raw and debiased leaderboards
- Run task-specific comparisons
- Start writing methods and results

### Week 5
- Run LLM-as-a-Judge comparison
- Add robustness checks
- Refine figures and paper draft

### Week 6
- Optional controlled experiment
- Final writing, editing, and presentation preparation

## Recommended Final Story for the Paper
The cleanest narrative is:

1. Arena voting is widely treated as gold-standard evaluation.
2. But cognitive science suggests preferences are constructed and context-sensitive.
3. Arena votes therefore likely combine model quality with systematic human bias.
4. A cognitive model can estimate those bias terms explicitly.
5. Removing those bias terms changes the resulting leaderboard and changes how we interpret LLM evaluation.

## Immediate Next Steps
1. Confirm the exact fields available in the Arena dataset.
2. Decide on the first task split, ideally creative versus factual/reasoning.
3. Implement a baseline Bradley-Terry style analysis.
4. Add the first bias terms: length and position.
5. Use those initial results to shape the final paper and decide whether the optional experiment is worth adding.
