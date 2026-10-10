import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

print("Loading embeddings...")
embeddings = np.load("data/processed/gnn_embeddings.npy")

print("Embeddings shape:", embeddings.shape)

print("Running Isolation Forest...")
iso = IsolationForest(
    n_estimators=100,
    contamination=0.01,   # assume top 1% suspicious
    random_state=42
)

preds = iso.fit_predict(embeddings)
scores = iso.decision_function(embeddings)

# Convert to anomaly score
anomaly_scores = -scores

print("Loading account names...")
df = pd.read_csv("data/processed/node_features.csv")

results = pd.DataFrame({
    "account": df["account"],
    "anomaly_score": anomaly_scores,
    "is_anomaly": (preds == -1).astype(int)
})

results = results.sort_values("anomaly_score", ascending=False)

results.to_csv("data/processed/anomaly_scores.csv", index=False)

print(results.head(20))
print("Saved anomaly scores.")