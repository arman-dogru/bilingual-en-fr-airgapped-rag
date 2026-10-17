## Retrieval (n=52 questions per direction; 95% Wilson CI on passage R@1)

| Direction | Reranker | passage R@1 | 95% CI | R@3 | R@5 | doc R@1 | doc R@5 |
|---|---|---|---|---|---|---|---|
| EN->EN | no | 0.904 | [0.79, 0.96] | 0.962 | 0.981 | 0.981 | 1.000 |
| EN->FR | no | 0.615 | [0.48, 0.73] | 0.846 | 0.923 | 0.731 | 0.942 |
| EN->MIXED | no | 0.904 | [0.79, 0.96] | 0.962 | 0.981 | 0.981 | 1.000 |
| FR->EN | no | 0.615 | [0.48, 0.73] | 0.904 | 0.962 | 0.808 | 0.981 |
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
| A_1.5b_same_basic | 52 | 71 | 0.47 | 0.366 | [0.244, 0.516] | 0.684 | 0.732 | [0.609, 0.851] | 0.94 | 0.45 / 0.30 |
| B_1.5b_same_fewshot | 52 | 58 | 0.12 | 0.586 | [0.435, 0.75] | 0.667 | 0.776 | [0.655, 0.891] | 1.00 | 0.61 / 0.56 |
| H_1.5b_same_fewshot_posthoc | 52 | 58 | 0.00 | 0.655 | [0.5, 0.79] | 0.655 | 0.776 | [0.655, 0.891] | 1.00 | 0.61 / 0.69 |
| D_0.5b_same_basic | 52 | 60 | 0.98 | 0.000 | [0.0, 0.0] | 0.000 | 0.683 | [0.527, 0.821] | 0.92 | 0.00 / 0.00 |
| E_0.5b_same_fewshot | 52 | 80 | 0.51 | 0.300 | [0.185, 0.434] | 0.615 | 0.787 | [0.687, 0.877] | 0.98 | 0.18 / 0.48 |
| F_0.5b_mixrr_fewshot | 52 | 89 | 0.65 | 0.202 | [0.11, 0.333] | 0.581 | 0.742 | [0.626, 0.849] | 0.98 | 0.17 / 0.26 |
| G_0.5b_mixrr_fewshot_posthoc | 52 | 89 | 0.00 | 0.573 | [0.465, 0.682] | 0.573 | 0.742 | [0.626, 0.849] | 0.98 | 0.59 / 0.54 |

## Generation and citation audit - DEV split (q01-q26, used to develop prompts/parser/threshold); 95% CI = bootstrap clustered by question

| Config | Answers | Claims | Uncited | Citation faithfulness | 95% CI | Faithful when cited | Grounded (any source) | 95% CI | Lang match | EN / FR |
|---|---|---|---|---|---|---|---|---|---|---|
| A_1.5b_same_basic | 52 | 85 | 0.48 | 0.318 | [0.232, 0.418] | 0.614 | 0.671 | [0.554, 0.778] | 1.00 | 0.48 / 0.22 |
| B_1.5b_same_fewshot | 52 | 72 | 0.29 | 0.569 | [0.442, 0.734] | 0.804 | 0.806 | [0.705, 0.887] | 1.00 | 0.65 / 0.50 |
| H_1.5b_same_fewshot_posthoc | 52 | 72 | 0.00 | 0.736 | [0.643, 0.831] | 0.736 | 0.806 | [0.705, 0.887] | 1.00 | 0.71 / 0.76 |
| D_0.5b_same_basic | 52 | 84 | 1.00 | 0.000 | [0.0, 0.0] | 0.000 | 0.583 | [0.412, 0.77] | 1.00 | 0.00 / 0.00 |
| E_0.5b_same_fewshot | 52 | 106 | 0.72 | 0.189 | [0.109, 0.296] | 0.667 | 0.792 | [0.689, 0.885] | 1.00 | 0.16 / 0.22 |
| F_0.5b_mixrr_fewshot | 52 | 105 | 0.73 | 0.152 | [0.086, 0.247] | 0.571 | 0.743 | [0.641, 0.842] | 1.00 | 0.08 / 0.26 |
| G_0.5b_mixrr_fewshot_posthoc | 52 | 105 | 0.00 | 0.581 | [0.477, 0.676] | 0.581 | 0.743 | [0.641, 0.842] | 1.00 | 0.54 / 0.64 |

## Paired deltas (citation faithfulness; paired bootstrap over questions, 5000 draws)

| Comparison | What | Split | Delta | 95% CI | p | CI excludes 0 |
|---|---|---|---|---|---|---|
| B-A | few-shot vs basic prompt (1.5B) | dev+test | +0.237 | [0.147, 0.333] | 0.0 | yes |
| B-A | few-shot vs basic prompt (1.5B) | test | +0.220 | [0.076, 0.372] | 0.0016 | yes |
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
| D_0.5b_same_basic | 4 -> 4 | 104/104 | 1.000 |
| D_0.5b_same_basic_threads8 | 4 -> 8 | 104/104 | 1.000 |

## Abstention on TEST (13 unanswerable x 2 langs vs 26 answerable x 2 langs; tau tuned on dev)

| Config | Mechanism | Precision | Recall | False-refusal rate | Unanswerable answered |
|---|---|---|---|---|---|
| A_1.5b_same_basic | llm | 0.67 | 0.85 | 0.21 | 4/26 |
| A_1.5b_same_basic | gate (tau=0.886, e5 cosine) | 0.68 | 1.00 | 0.23 | 0/26 |
| A_1.5b_same_basic | gate_or_llm (tau=0.886, e5 cosine) | 0.59 | 1.00 | 0.35 | 0/26 |
| B_1.5b_same_fewshot | llm | 1.00 | 0.12 | 0.00 | 23/26 |
| B_1.5b_same_fewshot | gate (tau=0.886, e5 cosine) | 0.68 | 1.00 | 0.23 | 0/26 |
| B_1.5b_same_fewshot | gate_or_llm (tau=0.886, e5 cosine) | 0.68 | 1.00 | 0.23 | 0/26 |
| D_0.5b_same_basic | llm | 0.50 | 0.50 | 0.25 | 13/26 |
| D_0.5b_same_basic | gate (tau=0.886, e5 cosine) | 0.68 | 1.00 | 0.23 | 0/26 |
| D_0.5b_same_basic | gate_or_llm (tau=0.886, e5 cosine) | 0.55 | 1.00 | 0.40 | 0/26 |
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

## NLI judge calibration (labels by the Developer agent, single annotator, dev questions; counts {'generated_neg': 5, 'generated_pos': 16, 'perturbed_neg': 16, 'source_swap_neg': 16})

| Setting | n | Acc [95% CI] | Precision | Recall [95% CI] | Specificity [95% CI] |
|---|---|---|---|---|---|
| raw@0.3/all | 53 | 0.849 [0.729, 0.921] | 0.682 | 0.938 [0.717, 0.989] | 0.811 [0.658, 0.905] |
| raw@0.5/generated_only | 21 | 0.857 [0.654, 0.95] | 0.882 | 0.938 [0.717, 0.989] | 0.6 [0.231, 0.882] |
| raw@0.5/positives+perturbed | 37 | 0.919 [0.787, 0.972] | 0.882 | 0.938 [0.717, 0.989] | 0.905 [0.711, 0.973] |
| raw@0.5/all | 53 | 0.887 [0.774, 0.947] | 0.75 | 0.938 [0.717, 0.989] | 0.865 [0.72, 0.941] |
| raw@0.7/all | 53 | 0.868 [0.752, 0.935] | 0.846 | 0.688 [0.444, 0.858] | 0.946 [0.823, 0.985] |
| raw@0.9/all | 53 | 0.849 [0.729, 0.921] | 1.0 | 0.5 [0.28, 0.72] | 1.0 [0.906, 1.0] |
| normalized@0.3/all | 53 | 0.849 [0.729, 0.921] | 0.667 | 1.0 [0.806, 1.0] | 0.784 [0.628, 0.886] |
| normalized@0.5/generated_only | 21 | 0.905 [0.711, 0.973] | 0.889 | 1.0 [0.806, 1.0] | 0.6 [0.231, 0.882] |
| normalized@0.5/positives+perturbed | 37 | 0.946 [0.823, 0.985] | 0.889 | 1.0 [0.806, 1.0] | 0.905 [0.711, 0.973] |
| normalized@0.5/all | 53 | 0.906 [0.797, 0.959] | 0.762 | 1.0 [0.806, 1.0] | 0.865 [0.72, 0.941] |
| normalized@0.7/all | 53 | 0.868 [0.752, 0.935] | 0.765 | 0.812 [0.57, 0.934] | 0.892 [0.753, 0.957] |
| normalized@0.9/all | 53 | 0.849 [0.729, 0.921] | 0.9 | 0.562 [0.332, 0.769] | 0.973 [0.862, 0.995] |
