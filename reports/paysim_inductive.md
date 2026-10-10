# PaySim — inductive account-risk model, without label leakage

Task: predict which accounts receive fraud-labelled money. Features are structure and amounts only (`src/features.py`), standardised per graph. Each time window is a separate graph, so the test graphs are entirely unseen during training. Scores are over receiving accounts; models are averaged over 3 seeds.

## Test window (steps 239-306)

Base rate: 0.110% of receiving accounts.

| Model | PR-AUC | ROC-AUC | Precision@100 | Precision@1000 |
|---|---|---|---|---|
| Largest receipt (1 feature) | 0.011 ± 0.000 | 0.848 | 0.00 | 0.01 |
| XGBoost (no graph) | 0.016 ± 0.000 | 0.834 | 0.15 | 0.05 |
| GraphSAGE (inductive) | 0.021 ± 0.002 | 0.847 | 0.15 | 0.05 |

## Late window (steps 373-743), where fraud is far more common

Base rate: 0.527% of receiving accounts.

| Model | PR-AUC | ROC-AUC | Precision@100 | Precision@1000 |
|---|---|---|---|---|
| Largest receipt (1 feature) | 0.099 ± 0.000 | 0.833 | 0.36 | 0.35 |
| XGBoost (no graph) | 0.094 ± 0.004 | 0.824 | 0.80 | 0.33 |
| GraphSAGE (inductive) | 0.132 ± 0.003 | 0.834 | 0.85 | 0.38 |
