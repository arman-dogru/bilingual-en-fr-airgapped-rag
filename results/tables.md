## Retrieval (n=52 questions per direction; 95% Wilson CI on passage R@1)

| Direction | Reranker | passage R@1 | 95% CI | R@3 | R@5 | doc R@1 | doc R@5 |
|---|---|---|---|---|---|---|---|
| EN->EN | no | 0.904 | [0.79, 0.96] | 0.962 | 0.981 | 0.981 | 1.000 |
| EN->FR | no | 0.615 | [0.48, 0.74] | 0.846 | 0.923 | 0.731 | 0.942 |
| EN->MIXED | no | 0.904 | [0.79, 0.96] | 0.962 | 0.981 | 0.981 | 1.000 |
| FR->EN | no | 0.615 | [0.48, 0.74] | 0.904 | 0.962 | 0.808 | 0.981 |
| FR->FR | no | 0.885 | [0.77, 0.95] | 1.000 | 1.000 | 0.962 | 1.000 |
| FR->MIXED | no | 0.885 | [0.77, 0.95] | 1.000 | 1.000 | 0.962 | 1.000 |
| EN->EN | yes | 0.904 | [0.79, 0.96] | 0.962 | 1.000 | 0.962 | 1.000 |
| EN->FR | yes | 0.769 | [0.64, 0.86] | 0.923 | 0.923 | 0.865 | 0.942 |
| EN->MIXED | yes | 0.904 | [0.79, 0.96] | 0.962 | 1.000 | 0.962 | 1.000 |
| FR->EN | yes | 0.904 | [0.79, 0.96] | 0.962 | 0.962 | 0.923 | 0.962 |
| FR->FR | yes | 0.885 | [0.77, 0.95] | 0.962 | 0.981 | 0.942 | 1.000 |
| FR->MIXED | yes | 0.885 | [0.77, 0.95] | 0.962 | 0.981 | 0.942 | 1.000 |

## Generation and citation audit - TEST split (q27-q52, never used for development); 95% CI = bootstrap clustered by question

| Config | Answers | Claims | Uncited | Citation faithfulness | 95% CI | Faithful when cited | Grounded (any source) | 95% CI | Lang match | EN / FR |
|---|---|---|---|---|---|---|---|---|---|---|
| A_1.5b_same_basic | 52 | 61 | 0.38 | 0.426 | [0.271, 0.623] | 0.684 | 0.787 | [0.652, 0.918] | 0.94 | 0.52 / 0.35 |
| B_1.5b_same_fewshot | 52 | 58 | 0.12 | 0.586 | [0.435, 0.75] | 0.667 | 0.776 | [0.655, 0.891] | 1.00 | 0.61 / 0.56 |
| H_1.5b_same_fewshot_posthoc | 52 | 58 | 0.00 | 0.655 | [0.5, 0.79] | 0.655 | 0.776 | [0.655, 0.891] | 1.00 | 0.61 / 0.69 |
| D_0.5b_same_basic | 52 | 48 | 0.98 | 0.000 | [0.0, 0.0] | 0.000 | 0.812 | [0.667, 0.933] | 0.92 | 0.00 / 0.00 |
| E_0.5b_same_fewshot | 52 | 80 | 0.51 | 0.300 | [0.185, 0.434] | 0.615 | 0.787 | [0.687, 0.877] | 0.98 | 0.18 / 0.48 |
| F_0.5b_mixrr_fewshot | 52 | 89 | 0.65 | 0.202 | [0.11, 0.333] | 0.581 | 0.742 | [0.626, 0.849] | 0.98 | 0.17 / 0.26 |
| G_0.5b_mixrr_fewshot_posthoc | 52 | 89 | 0.00 | 0.573 | [0.465, 0.682] | 0.573 | 0.742 | [0.626, 0.849] | 0.98 | 0.59 / 0.54 |

## Generation and citation audit - DEV split (q01-q26, used to develop prompts/parser/threshold); 95% CI = bootstrap clustered by question

| Config | Answers | Claims | Uncited | Citation faithfulness | 95% CI | Faithful when cited | Grounded (any source) | 95% CI | Lang match | EN / FR |
|---|---|---|---|---|---|---|---|---|---|---|
| A_1.5b_same_basic | 52 | 81 | 0.46 | 0.333 | [0.242, 0.443] | 0.614 | 0.691 | [0.57, 0.8] | 1.00 | 0.50 / 0.23 |
| B_1.5b_same_fewshot | 52 | 72 | 0.29 | 0.569 | [0.442, 0.734] | 0.804 | 0.806 | [0.705, 0.887] | 1.00 | 0.65 / 0.50 |
| H_1.5b_same_fewshot_posthoc | 52 | 72 | 0.00 | 0.736 | [0.643, 0.831] | 0.736 | 0.806 | [0.705, 0.887] | 1.00 | 0.71 / 0.76 |
| D_0.5b_same_basic | 52 | 79 | 1.00 | 0.000 | [0.0, 0.0] | 0.000 | 0.582 | [0.403, 0.779] | 1.00 | 0.00 / 0.00 |
| E_0.5b_same_fewshot | 52 | 106 | 0.72 | 0.189 | [0.109, 0.296] | 0.667 | 0.792 | [0.689, 0.885] | 1.00 | 0.16 / 0.22 |
| F_0.5b_mixrr_fewshot | 52 | 105 | 0.73 | 0.152 | [0.086, 0.247] | 0.571 | 0.743 | [0.641, 0.842] | 1.00 | 0.08 / 0.26 |
| G_0.5b_mixrr_fewshot_posthoc | 52 | 105 | 0.00 | 0.581 | [0.477, 0.676] | 0.581 | 0.743 | [0.641, 0.842] | 1.00 | 0.54 / 0.64 |

## Paired deltas (citation faithfulness; paired bootstrap over questions, 5000 draws)

| Comparison | What | Split | Delta | 95% CI | p | CI excludes 0 |
|---|---|---|---|---|---|---|
| B-A | few-shot vs basic prompt (1.5B) | dev+test | +0.204 | [0.108, 0.299] | 0.0 | yes |
| B-A | few-shot vs basic prompt (1.5B) | test | +0.160 | [-0.008, 0.321] | 0.0628 | no |
| E-D | few-shot vs basic prompt (0.5B) | dev+test | +0.237 | [0.164, 0.326] | 0.0 | yes |
| E-D | few-shot vs basic prompt (0.5B) | test | +0.300 | [0.185, 0.437] | 0.0 | yes |
| F-E | mixed-language + rerank vs same-language retrieval (0.5B) | dev+test | -0.061 | [-0.123, -0.005] | 0.0336 | yes |
| F-E | mixed-language + rerank vs same-language retrieval (0.5B) | test | -0.098 | [-0.206, 0.014] | 0.0852 | no |
| G-F | post-hoc attribution (0.5B) | dev+test | +0.402 | [0.321, 0.479] | 0.0 | yes |
| G-F | post-hoc attribution (0.5B) | test | +0.371 | [0.26, 0.48] | 0.0 | yes |
| H-B | post-hoc attribution (1.5B) | dev+test | +0.123 | [0.052, 0.194] | 0.0004 | yes |
| H-B | post-hoc attribution (1.5B) | test | +0.069 | [0.0, 0.169] | 0.2636 | no |
| B-E | 1.5B vs 0.5B, same retrieval + prompt | dev+test | +0.340 | [0.257, 0.428] | 0.0 | yes |
| B-E | 1.5B vs 0.5B, same retrieval + prompt | test | +0.286 | [0.179, 0.401] | 0.0 | yes |
| G-B | 0.5B+post-hoc vs 1.5B without (not like-for-like) | dev+test | +0.000 | [-0.119, 0.112] | 0.9808 | no |
| G-B | 0.5B+post-hoc vs 1.5B without (not like-for-like) | test | -0.013 | [-0.153, 0.114] | 0.8412 | no |
| G-H | 0.5B vs 1.5B, both post-hoc | dev+test | -0.123 | [-0.204, -0.044] | 0.0012 | yes |
| G-H | 0.5B vs 1.5B, both post-hoc | test | -0.082 | [-0.205, 0.042] | 0.1892 | no |

## Run-to-run determinism (byte-identical answers on dev+test)

| Rerun | Threads (ref -> rerun) | Identical | Match rate |
|---|---|---|---|
| B_1.5b_same_fewshot | 4 -> 4 | 99/104 | 0.952 |
| D_0.5b_same_basic | 4 -> 4 | 104/104 | 1.000 |
| D_0.5b_same_basic_threads8 | 4 -> 8 | 104/104 | 1.000 |

## Abstention on TEST (13 unanswerable x 2 langs vs 26 answerable x 2 langs; tau tuned on dev)

| Config | Mechanism | Precision | Recall | False-refusal rate | Unanswerable answered |
|---|---|---|---|---|---|
| A_1.5b_same_basic | llm | 0.69 | 0.85 | 0.19 | 4/26 |
| A_1.5b_same_basic | gate (tau=0.886, e5 cosine) | 0.68 | 1.00 | 0.23 | 0/26 |
| A_1.5b_same_basic | gate_or_llm (tau=0.886, e5 cosine) | 0.60 | 1.00 | 0.33 | 0/26 |
| B_1.5b_same_fewshot | llm | 1.00 | 0.12 | 0.00 | 23/26 |
| B_1.5b_same_fewshot | gate (tau=0.886, e5 cosine) | 0.68 | 1.00 | 0.23 | 0/26 |
| B_1.5b_same_fewshot | gate_or_llm (tau=0.886, e5 cosine) | 0.68 | 1.00 | 0.23 | 0/26 |
| D_0.5b_same_basic | llm | 0.52 | 0.50 | 0.23 | 13/26 |
| D_0.5b_same_basic | gate (tau=0.886, e5 cosine) | 0.68 | 1.00 | 0.23 | 0/26 |
| D_0.5b_same_basic | gate_or_llm (tau=0.886, e5 cosine) | 0.56 | 1.00 | 0.39 | 0/26 |
| E_0.5b_same_fewshot | llm | 0.00 | 0.00 | 0.00 | 26/26 |
| E_0.5b_same_fewshot | gate (tau=0.886, e5 cosine) | 0.68 | 1.00 | 0.23 | 0/26 |
| E_0.5b_same_fewshot | gate_or_llm (tau=0.886, e5 cosine) | 0.68 | 1.00 | 0.23 | 0/26 |
| F_0.5b_mixrr_fewshot | llm | 0.00 | 0.00 | 0.00 | 26/26 |
| F_0.5b_mixrr_fewshot | gate (tau=2.274, reranker logit) | 0.77 | 1.00 | 0.15 | 0/26 |
| F_0.5b_mixrr_fewshot | gate_or_llm (tau=2.274, reranker logit) | 0.77 | 1.00 | 0.15 | 0/26 |
| G_0.5b_mixrr_fewshot_posthoc | llm | 0.00 | 0.00 | 0.00 | 26/26 |
| G_0.5b_mixrr_fewshot_posthoc | gate (tau=2.274, reranker logit) | 0.77 | 1.00 | 0.15 | 0/26 |
| G_0.5b_mixrr_fewshot_posthoc | gate_or_llm (tau=2.274, reranker logit) | 0.77 | 1.00 | 0.15 | 0/26 |
| H_1.5b_same_fewshot_posthoc | llm | 1.00 | 0.12 | 0.00 | 23/26 |
| H_1.5b_same_fewshot_posthoc | gate (tau=0.886, e5 cosine) | 0.68 | 1.00 | 0.23 | 0/26 |
| H_1.5b_same_fewshot_posthoc | gate_or_llm (tau=0.886, e5 cosine) | 0.68 | 1.00 | 0.23 | 0/26 |

## NLI judge calibration (labels by the Developer agent, single annotator, dev questions; counts {'generated_neg': 5, 'generated_pos': 16, 'perturbed_neg': 16, 'source_swap_neg': 16, 'distractor_number_neg': 14})

| Setting | n | Acc [95% CI] | Precision | Recall [95% CI] | Specificity [95% CI] |
|---|---|---|---|---|---|
| cited/raw@0.3/all | 38 | 0.816 [0.666, 0.908] | 0.444 | 0.667 [0.3, 0.903] | 0.844 [0.682, 0.931] |
| cited/raw@0.5/generated_only | 10 | 0.7 [0.397, 0.892] | 0.8 | 0.667 [0.3, 0.903] | 0.75 [0.301, 0.954] |
| cited/raw@0.5/positives+perturbed | 17 | 0.824 [0.59, 0.938] | 0.8 | 0.667 [0.3, 0.903] | 0.909 [0.623, 0.984] |
| cited/raw@0.5/positives+distractor_number | 20 | 0.75 [0.531, 0.888] | 0.571 | 0.667 [0.3, 0.903] | 0.786 [0.524, 0.924] |
| cited/raw@0.5/all | 38 | 0.842 [0.696, 0.926] | 0.5 | 0.667 [0.3, 0.903] | 0.875 [0.719, 0.95] |
| cited/raw@0.7/all | 38 | 0.895 [0.759, 0.958] | 0.667 | 0.667 [0.3, 0.903] | 0.938 [0.799, 0.983] |
| cited/raw@0.9/all | 38 | 0.895 [0.759, 0.958] | 0.75 | 0.5 [0.188, 0.812] | 0.969 [0.843, 0.994] |
| any/raw@0.3/all | 67 | 0.776 [0.663, 0.859] | 0.517 | 0.938 [0.717, 0.989] | 0.725 [0.591, 0.829] |
| any/raw@0.5/generated_only | 21 | 0.857 [0.654, 0.95] | 0.882 | 0.938 [0.717, 0.989] | 0.6 [0.231, 0.882] |
| any/raw@0.5/positives+perturbed | 37 | 0.919 [0.787, 0.972] | 0.882 | 0.938 [0.717, 0.989] | 0.905 [0.711, 0.973] |
| any/raw@0.5/positives+distractor_number | 29 | 0.759 [0.579, 0.878] | 0.7 | 0.933 [0.702, 0.988] | 0.571 [0.326, 0.786] |
| any/raw@0.5/all | 67 | 0.821 [0.713, 0.894] | 0.577 | 0.938 [0.717, 0.989] | 0.784 [0.654, 0.875] |
| any/raw@0.7/all | 67 | 0.821 [0.713, 0.894] | 0.611 | 0.688 [0.444, 0.858] | 0.863 [0.743, 0.932] |
| any/raw@0.9/all | 67 | 0.866 [0.764, 0.928] | 0.889 | 0.5 [0.28, 0.72] | 0.98 [0.897, 0.997] |
| cited/normalized@0.3/all | 38 | 0.868 [0.727, 0.942] | 0.545 | 1.0 [0.61, 1.0] | 0.844 [0.682, 0.931] |
| cited/normalized@0.5/generated_only | 10 | 0.8 [0.49, 0.943] | 0.833 | 0.833 [0.436, 0.97] | 0.75 [0.301, 0.954] |
| cited/normalized@0.5/positives+perturbed | 17 | 0.882 [0.657, 0.967] | 0.833 | 0.833 [0.436, 0.97] | 0.909 [0.623, 0.984] |
| cited/normalized@0.5/positives+distractor_number | 20 | 0.8 [0.584, 0.919] | 0.625 | 0.833 [0.436, 0.97] | 0.786 [0.524, 0.924] |
| cited/normalized@0.5/all | 38 | 0.842 [0.696, 0.926] | 0.5 | 0.833 [0.436, 0.97] | 0.844 [0.682, 0.931] |
| cited/normalized@0.7/all | 38 | 0.868 [0.727, 0.942] | 0.556 | 0.833 [0.436, 0.97] | 0.875 [0.719, 0.95] |
| cited/normalized@0.9/all | 38 | 0.895 [0.759, 0.958] | 0.667 | 0.667 [0.3, 0.903] | 0.938 [0.799, 0.983] |
| any/normalized@0.3/all | 67 | 0.776 [0.663, 0.859] | 0.516 | 1.0 [0.806, 1.0] | 0.706 [0.57, 0.813] |
| any/normalized@0.5/generated_only | 21 | 0.905 [0.711, 0.973] | 0.889 | 1.0 [0.806, 1.0] | 0.6 [0.231, 0.882] |
| any/normalized@0.5/positives+perturbed | 37 | 0.946 [0.823, 0.985] | 0.889 | 1.0 [0.806, 1.0] | 0.905 [0.711, 0.973] |
| any/normalized@0.5/positives+distractor_number | 29 | 0.793 [0.616, 0.902] | 0.714 | 1.0 [0.796, 1.0] | 0.571 [0.326, 0.786] |
| any/normalized@0.5/all | 67 | 0.836 [0.729, 0.906] | 0.593 | 1.0 [0.806, 1.0] | 0.784 [0.654, 0.875] |
| any/normalized@0.7/all | 67 | 0.821 [0.713, 0.894] | 0.591 | 0.812 [0.57, 0.934] | 0.824 [0.697, 0.904] |
| any/normalized@0.9/all | 67 | 0.866 [0.764, 0.928] | 0.818 | 0.562 [0.332, 0.769] | 0.961 [0.868, 0.989] |

## Gate threshold stability (1000 bootstrap resamples of DEV; percentiles 2.5 / 50 / 97.5)

| Gate | Configs | tau (point) | tau pct | TEST recall pct | TEST false-refusal pct | TEST-oracle tau |
|---|---|---|---|---|---|---|
| e5 cosine | A, B, D, E, H | 0.886 | 0.864 / 0.886 / 0.890 | 1.000 / 1.000 / 1.000 | 0.038 / 0.231 / 0.269 | 0.867 |
| reranker logit | F, G | 2.274 | -0.877 / -0.146 / 4.780 | 0.962 / 1.000 / 1.000 | 0.019 / 0.058 / 0.269 | -0.314 |

## Robustness (F_0.5b_mixrr_fewshot, tau=2.274; Wilson 95% CI, rows clustered by question so CIs are optimistic)

### In-domain unanswerable questions (data/unanswerable_indomain.json)

| Set | n | Gate recall | LLM refusal recall | Gate-or-LLM recall | top reranker score - tau (min, max) |
|---|---|---|---|---|---|
| v01-v20 (off-target) | 40 | 1.00 (40/40) [0.912, 1] | 0.03 (1/40) [0.004, 0.129] | 1.00 (40/40) [0.912, 1] | [-10.43, -2.35] |
| w01-w08 (near-miss) | 16 | 1.00 (16/16) [0.806, 1] | 0.00 (0/16) [0, 0.194] | 1.00 (16/16) [0.806, 1] | [-8.12, -2.01] |

### Prompt injection (data/injection.json; success = injected marker in the answer; strict = src.ask final answer)

| Site / template | n (questions) | Success raw | After gate | After gate + strict | Gated | Strict refuses non-hijacked |
|---|---|---|---|---|---|---|
| poisoned_chunk/ignore | 20 (10) | 0.00 (0/20) [0, 0.161] | 0.00 (0/20) [0, 0.161] | 0.00 (0/20) [0, 0.161] | 0.15 (3/20) [0.052, 0.36] | 0.59 (10/17) [0.36, 0.784] |
| poisoned_chunk/cite | 20 (10) | 0.00 (0/20) [0, 0.161] | 0.00 (0/20) [0, 0.161] | 0.00 (0/20) [0, 0.161] | 0.15 (3/20) [0.052, 0.36] | 0.77 (13/17) [0.527, 0.904] |
| poisoned_chunk/refusal_prefix | 20 (10) | 0.00 (0/20) [0, 0.161] | 0.00 (0/20) [0, 0.161] | 0.00 (0/20) [0, 0.161] | 0.15 (3/20) [0.052, 0.36] | 0.71 (12/17) [0.469, 0.867] |
| poisoned_chunk/all | 60 (10) | 0.00 (0/60) [0, 0.06] | 0.00 (0/60) [0, 0.06] | 0.00 (0/60) [0, 0.06] | 0.15 (9/60) [0.081, 0.261] | 0.69 (35/51) [0.55, 0.797] |
| query_side/ignore | 20 (10) | 0.65 (13/20) [0.433, 0.819] | 0.55 (11/20) [0.342, 0.742] | 0.00 (0/20) [0, 0.161] | 0.20 (4/20) [0.081, 0.416] | 0.80 (4/5) [0.376, 0.964] |
| query_side/cite | 20 (10) | 0.25 (5/20) [0.112, 0.469] | 0.25 (5/20) [0.112, 0.469] | 0.00 (0/20) [0, 0.161] | 0.20 (4/20) [0.081, 0.416] | 0.55 (6/11) [0.28, 0.787] |
| query_side/refusal_prefix | 20 (10) | 0.45 (9/20) [0.258, 0.658] | 0.35 (7/20) [0.181, 0.567] | 0.00 (0/20) [0, 0.161] | 0.15 (3/20) [0.052, 0.36] | 0.90 (9/10) [0.596, 0.982] |
| query_side/all | 60 (10) | 0.45 (27/60) [0.331, 0.575] | 0.38 (23/60) [0.271, 0.51] | 0.00 (0/60) [0, 0.06] | 0.18 (11/60) [0.106, 0.299] | 0.73 (19/26) [0.539, 0.863] |

### Cost of strict mode on TEST answers (config F, committed answers)

| n | Gated | Raw refusal (ungated) | Strict -> refusal (ungated) | Strict leaves answer unchanged (ungated) | Answered after gate + strict |
|---|---|---|---|---|---|
| 52 | 0.15 (8/52) [0.08, 0.275] | 0.00 (0/44) [0.0, 0.08] | 0.59 (26/44) [0.444, 0.723] | 0.39 (17/44) [0.257, 0.534] | 0.35 (18/52) [0.232, 0.482] |
