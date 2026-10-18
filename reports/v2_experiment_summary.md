# nano-CodeSelfPlay · v2 results

Twelve completed runs: four configurations × training seeds 42, 43, 44. Each model is selected by validation Greedy Pass@1, keeping the earliest tie.

Mean ± **sample SD (ddof=1)**. SD is not a confidence interval. The test benchmark contains 20 historically inspected problems; it is not a newly unseen blind set.

| Method | Validation best | Test greedy | Test Pass@4 | Candidate hit rate |
| :--- | ---: | ---: | ---: | ---: |
| Base | 55.00% | 40.00% | 55.00 ± 5.00% | 30.00 ± 3.31% |
| Static GRPO | 80.00 ± 0.00% | 63.33 ± 10.41% | 80.00 ± 0.00% | 52.50 ± 5.00% |
| Self-play | 80.00 ± 0.00% | 60.00 ± 5.00% | 80.00 ± 13.23% | 54.17 ± 8.04% |
| Mixed replay (configured ratio 0.1) | 83.33 ± 2.89% | 58.33 ± 2.89% | 80.00 ± 8.66% | 59.17 ± 7.11% |
| Frozen after warmup | 80.00 ± 8.66% | 58.33 ± 10.41% | 76.67 ± 7.64% | 51.67 ± 5.91% |

Base sampling seeds repeat decoding of one pretrained model; other rows vary training and final decoding. Mixed replay uses a configured selection probability of 0.1, not a measured 10% share.

## Training cost

75 outer steps each; static has 75 Breaker opportunities, other groups have 70 plus optional rescue. Not matched compute.

| Method | Breaker updates at last | Breaker candidates | Mutator candidates | Wall time (s) |
| :--- | ---: | ---: | ---: | ---: |
| Static GRPO | 43.00 ± 3.00 | 600.00 ± 0.00 | 0.00 ± 0.00 | 312.63 ± 55.43 |
| Self-play | 54.00 ± 3.61 | N/A (not recorded) | N/A (not recorded) | N/A (not recorded) |
| Mixed replay (configured ratio 0.1) | 53.67 ± 5.03 | N/A (not recorded) | N/A (not recorded) | N/A (not recorded) |
| Frozen after warmup | 59.33 ± 1.53 | N/A (not recorded) | N/A (not recorded) | N/A (not recorded) |

Update counts are checked against AdamW step counters in [checkpoint_audit.json](checkpoint_audit.json). Missing historical counters/time are N/A, not estimates. Static wall time covers the training loop, periodic validation and checkpoint saves; it excludes loading, the initial baseline audit and final test.

A frozen Mutator still generates new candidates throughout training. The configured 75 × 4 = 300 Mutator candidates must not be described as 20. Old runs did not record all Breaker probe/rescue generation counts.

## Per-seed results

### Static GRPO

| Seed | Selected step | Test greedy | Test Pass@4 | Hit rate | Updates: selected / last |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 42 | 45 | 55.00% | 80.00% | 47.50% | 28 / 46 |
| 43 | 60 | 60.00% | 80.00% | 57.50% | 34 / 40 |
| 44 | 45 | 75.00% | 80.00% | 52.50% | 30 / 43 |

### Self-play

| Seed | Selected step | Test greedy | Test Pass@4 | Hit rate | Updates: selected / last |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 42 | 45 | 60.00% | 85.00% | 60.00% | 35 / 57 |
| 43 | 45 | 65.00% | 90.00% | 57.50% | 35 / 55 |
| 44 | 45 | 55.00% | 65.00% | 45.00% | 33 / 50 |

### Mixed replay (configured ratio 0.1)

| Seed | Selected step | Test greedy | Test Pass@4 | Hit rate | Updates: selected / last |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 42 | 60 | 55.00% | 90.00% | 65.00% | 51 / 59 |
| 43 | 60 | 60.00% | 75.00% | 61.25% | 43 / 53 |
| 44 | 45 | 60.00% | 75.00% | 51.25% | 31 / 49 |

### Frozen after warmup

| Seed | Selected step | Test greedy | Test Pass@4 | Hit rate | Updates: selected / last |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 42 | 45 | 55.00% | 85.00% | 56.25% | 37 / 61 |
| 43 | 30 | 50.00% | 70.00% | 45.00% | 24 / 59 |
| 44 | 45 | 70.00% | 75.00% | 53.75% | 35 / 58 |

## Provenance

Raw final reports and validation trajectories are bundled in [runs/](runs/), including training/data fingerprints and decoding settings. The original model weights remain untracked. Rebuild this file with `python scripts/aggregate_all_reports.py`.
