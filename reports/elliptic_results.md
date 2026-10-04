# Elliptic benchmark — does the graph help?

Dataset: Elliptic Bitcoin (Weber et al., 2019) — 203,769 transactions, 234,355 payment-flow edges, 46,564 labelled (4,545 illicit).

Temporal split: train = time steps 1-29 (26,381 labelled), validation = 30-34 (3,513), test = 35-49 (16,670, 1,083 illicit). Thresholds are chosen on validation and frozen. All metrics are for the illicit class on the test period; neural and tree models are averaged over 3 seeds.

## All features (165)

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.171 | 0.780 | 0.281 ± 0.000 | 0.219 | 0.843 |
| Random Forest | 0.895 | 0.714 | **0.794 ± 0.016** | 0.787 | 0.929 |
| XGBoost | 0.781 | 0.735 | 0.756 ± 0.027 | 0.789 | 0.918 |
| MLP (no graph) | 0.488 | 0.480 | 0.482 ± 0.008 | 0.314 | 0.865 |
| GraphSAGE | 0.571 | 0.569 | 0.569 ± 0.016 | 0.518 | 0.870 |
| GraphSAGE emb. + XGBoost | 0.580 | 0.585 | 0.581 ± 0.002 | 0.617 | 0.892 |

## Local features only (93)

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.254 | 0.743 | 0.378 ± 0.000 | 0.171 | 0.817 |
| Random Forest | 0.884 | 0.711 | **0.788 ± 0.004** | 0.779 | 0.897 |
| XGBoost | 0.790 | 0.687 | 0.730 ± 0.026 | 0.771 | 0.911 |
| MLP (no graph) | 0.556 | 0.705 | 0.621 ± 0.020 | 0.451 | 0.892 |
| GraphSAGE | 0.603 | 0.512 | 0.554 ± 0.002 | 0.455 | 0.862 |
| GraphSAGE emb. + XGBoost | 0.535 | 0.538 | 0.536 ± 0.005 | 0.541 | 0.861 |

![F1 per time step](elliptic_f1_over_time.png)
