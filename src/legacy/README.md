# Legacy experiments

Earlier experiments that the app and the current training pipeline do not use.
They are kept for reference and are not maintained.

- `models/ann*.py`, `training/train_ann.py` — dense encoder on engineered account statistics
- `models/lstm*.py`, `training/train_lstm.py`, `preprocessing/create_sequence.py` — LSTM over transaction sequences
- `models/gnn.py`, `models/gnn_model.py` — first GraphSAGE drafts, replaced by `src/models/graphsage.py`
- `models/edge_decoder.py`, `training/train_edge_prediction.py` — alternative edge-prediction setup; the script no longer matches the encoder's arguments
- `inference/detect_anomalies.py`, `inference/cluster_fraud_rings.py` — Isolation Forest and DBSCAN over embeddings from the old, leaky pipeline
