"""
evaluate_elliptic.py
--------------------
Honest benchmark of the FRECTION GraphSAGE encoder on the Elliptic Bitcoin
dataset (real transactions, real illicit / licit labels), against non-graph
baselines.

Question this answers:  "Does the graph actually help, and by how much?"

Protocol (no leakage)
---------------------
* Temporal split — the model never sees the future:
      train  : time steps  1-29
      val    : time steps 30-34   (early stopping + decision threshold)
      test   : time steps 35-49   (reported numbers)
  Elliptic has no edges between time steps, so message passing cannot leak
  test information into training.
* Feature scaling is fitted on the training time steps only.
* The decision threshold of every model is chosen on validation (max F1) and
  then frozen — never tuned on test.
* Neural models are run with several seeds and reported as mean +/- std.

Models
------
  Logistic Regression, Random Forest, XGBoost      tabular baselines
  MLP                                              same features, no graph
  GraphSAGE                                        src.models.graphsage.GraphSAGEEncoder + linear head
  GraphSAGE embeddings + XGBoost                   hybrid

Each is evaluated with ALL 165 features and with the 93 LOCAL features only
(the remaining 72 are neighbour aggregates pre-computed by the dataset authors,
i.e. graph information already baked into the table).

Usage
-----
  python -m src.evaluation.evaluate_elliptic --data-dir data/raw/elleptical
Outputs
-------
  reports/elliptic_results.json, reports/elliptic_results.md,
  reports/elliptic_f1_over_time.png
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from xgboost import XGBClassifier

from src.models.graphsage import GraphSAGEEncoder

ROOT        = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
REPORTS_DIR = os.path.join(ROOT, "reports")

TRAIN_END, VAL_END = 29, 34      # inclusive time-step boundaries
N_LOCAL_FEATURES   = 93
SEEDS              = [0, 1, 2]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_elliptic(data_dir: str) -> dict:
    cache = os.path.join(ROOT, "data", "processed", "elliptic_cache.npz")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}

    print("Parsing Elliptic CSVs (first run only)...")
    feats   = pd.read_csv(os.path.join(data_dir, "elliptic_txs_features.csv"), header=None)
    classes = pd.read_csv(os.path.join(data_dir, "elliptic_txs_classes.csv"))
    edges   = pd.read_csv(os.path.join(data_dir, "elliptic_txs_edgelist.csv"))

    tx_ids = feats[0].to_numpy()
    index  = pd.Series(np.arange(len(tx_ids)), index=tx_ids)

    # 1 = illicit, 0 = licit, -1 = unknown
    label_map = {"1": 1, "2": 0, "unknown": -1}
    y = classes.set_index("txId")["class"].astype(str).map(label_map).reindex(tx_ids).to_numpy()

    data = {
        "x":          feats.iloc[:, 2:].to_numpy(dtype=np.float32),
        "time":       feats[1].to_numpy(dtype=np.int64),
        "y":          y.astype(np.int64),
        "edge_index": np.stack([index.loc[edges["txId1"]].to_numpy(), index.loc[edges["txId2"]].to_numpy()]),
    }
    np.savez(cache, **data)
    return data


def make_splits(d: dict) -> dict:
    labelled = d["y"] >= 0
    t = d["time"]
    return {
        "train": np.flatnonzero(labelled & (t <= TRAIN_END)),
        "val":   np.flatnonzero(labelled & (t > TRAIN_END) & (t <= VAL_END)),
        "test":  np.flatnonzero(labelled & (t > VAL_END)),
    }


def standardise(x: np.ndarray, time: np.ndarray) -> np.ndarray:
    fit_rows = time <= TRAIN_END                 # statistics from the training period only
    mean, std = x[fit_rows].mean(0), x[fit_rows].std(0)
    return ((x - mean) / np.where(std > 0, std, 1.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def best_f1_threshold(y_true, scores) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def evaluate(y_true, scores, threshold) -> dict:
    pred = scores >= threshold
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall":    float(recall_score(y_true, pred, zero_division=0)),
        "f1":        float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc":    float(average_precision_score(y_true, scores)),
        "roc_auc":   float(roc_auc_score(y_true, scores)),
    }


def f1_by_time_step(y_true, scores, threshold, time) -> dict:
    out = {}
    for step in np.unique(time):
        m = time == step
        if y_true[m].sum() > 0:
            out[int(step)] = float(f1_score(y_true[m], scores[m] >= threshold, zero_division=0))
    return out


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def tabular_models(pos_weight: float, seed: int) -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1),
        "Random Forest":       RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                      n_jobs=-1, random_state=seed),
        "XGBoost":             XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                                             colsample_bytree=0.8, scale_pos_weight=pos_weight,
                                             eval_metric="aucpr", n_jobs=-1, random_state=seed),
    }


class MLP(nn.Module):
    """Same capacity as the GNN, but every transaction is scored in isolation."""
    def __init__(self, in_dim, hidden=128, out_dim=64, dropout=0.3):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim))
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(out_dim, 1))

    def embed(self, x, edge_index):
        return self.body(x)

    def forward(self, x, edge_index):
        return self.head(self.embed(x, edge_index)).squeeze(-1)


class SAGEClassifier(nn.Module):
    """FRECTION's GraphSAGE encoder with a node-classification head."""
    def __init__(self, in_dim, hidden=128, out_dim=64, dropout=0.3):
        super().__init__()
        self.encoder = GraphSAGEEncoder(in_channels=in_dim, hidden_channels=hidden, out_channels=out_dim, dropout=dropout)
        self.head = nn.Sequential(nn.ReLU(), nn.Dropout(dropout), nn.Linear(out_dim, 1))

    def embed(self, x, edge_index):
        return self.encoder(x, edge_index)

    def forward(self, x, edge_index):
        return self.head(self.embed(x, edge_index)).squeeze(-1)


def train_neural(model_cls, x, edge_index, y, splits, seed, epochs=300, patience=40, lr=5e-3):
    """Full-batch training; early stopping on validation PR-AUC. Returns (scores for all nodes, embeddings)."""
    torch.manual_seed(seed)
    model = model_cls(x.size(1))
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    tr, va = torch.from_numpy(splits["train"]), splits["val"]
    y_tr   = y[tr].float()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum())
    y_va   = y[va].numpy()

    best, best_state, wait = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss_fn(model(x, edge_index)[tr], y_tr).backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_ap = average_precision_score(y_va, torch.sigmoid(model(x, edge_index)[va]).numpy())
        if val_ap > best:
            best, wait = val_ap, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(x, edge_index)).numpy()
        emb    = model.embed(x, edge_index).numpy()
    return scores, emb


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def summarise(runs: list[dict]) -> dict:
    keys = runs[0].keys()
    return {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))} for k in keys}


def run_feature_set(name, x_np, d, splits) -> tuple[dict, dict]:
    print(f"\n=== Feature set: {name} ({x_np.shape[1]} features) ===")
    y_np, time_np = d["y"], d["time"]
    tr, va, te = splits["train"], splits["val"], splits["test"]
    y_tr, y_va, y_te = y_np[tr], y_np[va], y_np[te]
    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())

    results, over_time = {}, {}

    def record(model_name, per_seed_scores):
        runs, curves = [], []
        for scores in per_seed_scores:
            thr = best_f1_threshold(y_va, scores[va])
            runs.append(evaluate(y_te, scores[te], thr))
            curves.append(f1_by_time_step(y_te, scores[te], thr, time_np[te]))
        results[model_name] = summarise(runs)
        over_time[model_name] = {s: float(np.mean([c[s] for c in curves])) for s in curves[0]}
        r = results[model_name]
        print(f"  {model_name:<28} F1 {r['f1']['mean']:.3f} ±{r['f1']['std']:.3f} | "
              f"P {r['precision']['mean']:.3f} R {r['recall']['mean']:.3f} | PR-AUC {r['pr_auc']['mean']:.3f}")

    # --- tabular baselines (deterministic given a seed; LR has no seed) ---
    for model_name in tabular_models(pos_weight, 0):
        seeds = [0] if model_name == "Logistic Regression" else SEEDS
        per_seed = []
        for seed in seeds:
            model = tabular_models(pos_weight, seed)[model_name]
            model.fit(x_np[tr], y_tr)
            scores = np.zeros(len(y_np), dtype=np.float32)
            idx = np.concatenate([va, te])
            scores[idx] = model.predict_proba(x_np[idx])[:, 1]
            per_seed.append(scores)
        record(model_name, per_seed)

    # --- neural models ---
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    edge_index = torch.from_numpy(d["edge_index"])
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)   # message passing in both directions

    record("MLP (no graph)", [train_neural(MLP, x, edge_index, y, splits, s)[0] for s in SEEDS])

    sage_runs = [train_neural(SAGEClassifier, x, edge_index, y, splits, s) for s in SEEDS]
    record("GraphSAGE", [scores for scores, _ in sage_runs])

    # --- hybrid: GraphSAGE embeddings appended to the raw features, classified by XGBoost ---
    per_seed = []
    for seed, (_, emb) in zip(SEEDS, sage_runs):
        xh = np.hstack([x_np, emb])
        model = tabular_models(pos_weight, seed)["XGBoost"]
        model.fit(xh[tr], y_tr)
        scores = np.zeros(len(y_np), dtype=np.float32)
        idx = np.concatenate([va, te])
        scores[idx] = model.predict_proba(xh[idx])[:, 1]
        per_seed.append(scores)
    record("GraphSAGE emb. + XGBoost", per_seed)

    return results, over_time


def write_reports(all_results, all_over_time, d, splits, elapsed):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    y = d["y"]
    meta = {
        "dataset": "Elliptic Bitcoin (Weber et al., 2019)",
        "nodes": int(len(y)), "edges": int(d["edge_index"].shape[1]),
        "labelled": int((y >= 0).sum()), "illicit": int((y == 1).sum()),
        "split": {k: {"nodes": int(len(v)), "illicit": int(y[v].sum())} for k, v in splits.items()},
        "protocol": f"train t<= {TRAIN_END}, val t<= {VAL_END}, test t> {VAL_END}; threshold chosen on val; seeds {SEEDS}",
        "runtime_seconds": round(elapsed, 1),
    }
    with open(os.path.join(REPORTS_DIR, "elliptic_results.json"), "w") as f:
        json.dump({"meta": meta, "results": all_results, "f1_over_time": all_over_time}, f, indent=2)

    lines = [
        "# Elliptic benchmark — does the graph help?", "",
        f"Dataset: {meta['dataset']} — {meta['nodes']:,} transactions, {meta['edges']:,} payment-flow edges, "
        f"{meta['labelled']:,} labelled ({meta['illicit']:,} illicit).", "",
        f"Temporal split: train = time steps 1-{TRAIN_END} ({meta['split']['train']['nodes']:,} labelled), "
        f"validation = {TRAIN_END + 1}-{VAL_END} ({meta['split']['val']['nodes']:,}), "
        f"test = {VAL_END + 1}-49 ({meta['split']['test']['nodes']:,}, {meta['split']['test']['illicit']:,} illicit). "
        "Thresholds are chosen on validation and frozen. All metrics are for the illicit class on the test period; "
        f"neural and tree models are averaged over {len(SEEDS)} seeds.", "",
    ]
    for fs, results in all_results.items():
        lines += [f"## {fs}", "", "| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |", "|---|---|---|---|---|---|"]
        best = max(results, key=lambda m: results[m]["f1"]["mean"])
        for model_name, r in results.items():
            f1 = f"{r['f1']['mean']:.3f} ± {r['f1']['std']:.3f}"
            f1 = f"**{f1}**" if model_name == best else f1
            lines.append(f"| {model_name} | {r['precision']['mean']:.3f} | {r['recall']['mean']:.3f} | {f1} | "
                         f"{r['pr_auc']['mean']:.3f} | {r['roc_auc']['mean']:.3f} |")
        lines.append("")
    lines += ["![F1 per time step](elliptic_f1_over_time.png)", ""]
    with open(os.path.join(REPORTS_DIR, "elliptic_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(all_over_time), figsize=(7 * len(all_over_time), 4), sharey=True)
    for ax, (fs, curves) in zip(np.atleast_1d(axes), all_over_time.items()):
        for model_name in ("XGBoost", "MLP (no graph)", "GraphSAGE", "GraphSAGE emb. + XGBoost"):
            steps = sorted(curves[model_name])
            ax.plot(steps, [curves[model_name][s] for s in steps], marker="o", ms=3, label=model_name)
        ax.axvline(43, color="grey", ls="--", lw=1)
        ax.text(43.2, 0.02, "dark-market\nshutdown", fontsize=8, color="grey")
        ax.set_title(fs)
        ax.set_xlabel("test time step")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("illicit F1")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "elliptic_f1_over_time.png"), dpi=140)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(ROOT, "data", "raw", "elleptical"))
    args = parser.parse_args()

    t0 = time.time()
    d = load_elliptic(args.data_dir)
    splits = make_splits(d)
    y = d["y"]
    print(f"Nodes {len(y):,} | edges {d['edge_index'].shape[1]:,} | labelled {(y >= 0).sum():,} | illicit {(y == 1).sum():,}")
    for k, v in splits.items():
        print(f"  {k:<5} {len(v):>6,} labelled, {int(y[v].sum()):>5,} illicit ({y[v].mean():.1%})")

    x_all = standardise(d["x"], d["time"])
    all_results, all_over_time = {}, {}
    for name, x in (("All features (165)", x_all), ("Local features only (93)", x_all[:, :N_LOCAL_FEATURES])):
        all_results[name], all_over_time[name] = run_feature_set(name, np.ascontiguousarray(x), d, splits)

    write_reports(all_results, all_over_time, d, splits, time.time() - t0)
    print(f"\nDone in {time.time() - t0:.0f}s — see reports/elliptic_results.md")


if __name__ == "__main__":
    main()
