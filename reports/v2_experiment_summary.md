# nano-CodeSelfPlay · v2 results

Twelve completed runs: four configurations × training seeds 42, 43, 44. Each model is selected by validation Greedy Pass@1, keeping the earliest tie.

Mean ± **sample SD (ddof=1)**. SD is not a confidence interval. The test benchmark contains 20 historically inspected problems; it is not a newly unseen blind set.

| Method | Validation best | Test greedy | Test Pass@4 | Candidate hit rate |
| :--- | ---: | ---: | ---: | ---: |
| Base | 55.00% | 40.00% | 55.00 ± 5.00% | 30.00 ± 3.31% |
| Static GRPO | 78.33 ± 2.89% | 65.00 ± 10.00% | 85.00 ± 10.00% | 61.67 ± 8.32% |
| Self-play | 96.67 ± 2.89% | 75.00 ± 5.00% | 96.67 ± 2.89% | 75.83 ± 2.89% |
| Mixed replay (configured ratio 0.1) | 96.67 ± 2.89% | 75.00 ± 5.00% | 96.67 ± 5.77% | 77.08 ± 6.41% |
| Frozen after warmup | 83.33 ± 2.89% | 66.67 ± 2.89% | 83.33 ± 10.41% | 58.75 ± 3.75% |

Base sampling seeds repeat decoding of one pretrained model; other rows vary training and final decoding. Mixed replay uses a configured selection probability of 0.1, not a measured 10% share.

## Training cost

75 outer steps each; static has 75 Breaker opportunities, other groups have 70 plus optional rescue. Not matched compute.

| Method | Breaker updates at last | Breaker candidates | Mutator candidates | Wall time (s) |
| :--- | ---: | ---: | ---: | ---: |
| Static GRPO | 41.67 ± 3.51 | 600.00 ± 0.00 | 0.00 ± 0.00 | 229.83 ± 22.21 |
| Self-play | 70.00 ± 0.00 | 3202.67 ± 22.74 | 300.00 ± 0.00 | 971.92 ± 20.38 |
| Mixed replay (configured ratio 0.1) | 70.00 ± 0.00 | 3225.33 ± 4.62 | 300.00 ± 0.00 | 1001.27 ± 26.43 |
| Frozen after warmup | 60.33 ± 1.15 | 1256.00 ± 36.00 | 300.00 ± 0.00 | 497.21 ± 7.90 |

Update counts are checked against AdamW step counters in [checkpoint_audit.json](checkpoint_audit.json). Missing historical counters/time are N/A, not estimates. Static wall time covers the training loop, periodic validation and checkpoint saves; it excludes loading, the initial baseline audit and final test.

A frozen Mutator still generates new candidates throughout training. The configured 75 × 4 = 300 Mutator candidates must not be described as 20. Old runs did not record all Breaker probe/rescue generation counts.

## Per-seed results

### Static GRPO

| Seed | Selected step | Test greedy | Test Pass@4 | Hit rate | Updates: selected / last |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 42 | 75 | 65.00% | 95.00% | 68.75% | 42 / 42 |
| 43 | 60 | 75.00% | 85.00% | 63.75% | 37 / 45 |
| 44 | 45 | 55.00% | 75.00% | 52.50% | 26 / 38 |

### Self-play

| Seed | Selected step | Test greedy | Test Pass@4 | Hit rate | Updates: selected / last |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 42 | 75 | 75.00% | 95.00% | 77.50% | 70 / 70 |
| 43 | 75 | 70.00% | 100.00% | 77.50% | 70 / 70 |
| 44 | 75 | 80.00% | 95.00% | 72.50% | 70 / 70 |

### Mixed replay (configured ratio 0.1)

| Seed | Selected step | Test greedy | Test Pass@4 | Hit rate | Updates: selected / last |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 42 | 75 | 80.00% | 100.00% | 82.50% | 70 / 70 |
| 43 | 75 | 75.00% | 100.00% | 78.75% | 70 / 70 |
| 44 | 75 | 70.00% | 90.00% | 70.00% | 70 / 70 |

### Frozen after warmup

| Seed | Selected step | Test greedy | Test Pass@4 | Hit rate | Updates: selected / last |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 42 | 60 | 70.00% | 95.00% | 62.50% | 50 / 61 |
| 43 | 45 | 65.00% | 75.00% | 55.00% | 34 / 59 |
| 44 | 60 | 65.00% | 80.00% | 58.75% | 48 / 61 |

## Provenance

Raw final reports and validation trajectories are bundled in [runs/](runs/), including training/data fingerprints and decoding settings. The original model weights remain untracked. Rebuild this file with `python scripts/aggregate_all_reports.py`.
