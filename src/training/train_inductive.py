"""
train_inductive.py
------------------
Trains the inductive account-risk model (GraphSAGE) on PaySim with NO label
leakage, and benchmarks it against XGBoost on the same features.

Task      : for each account, predict whether it receives fraud-labelled money.
Features  : src/features.py — structure and amounts only, standardised per graph.
Inductive : PaySim is cut into time windows and each window is its own graph.
            The model trains on early windows and is tested on later windows it
            has never seen — the same situation as a freshly uploaded ledger.

    train : rows 0.00M-2.12M   (two graphs, steps   1-183)
    val   : rows 2.12M-3.18M   (steps 183-239)   early stopping
    test  : rows 3.18M-4.24M   (steps 239-306)   reported
    late  : rows 5.30M-6.36M   (steps 373-743)   reported; fraud is far more common here

Usage
-----
  python -m src.training.train_inductive --paysim data/raw/paysim/paysim.csv
Outputs
-------
  models/graphsage_inductive.pt, reports/paysim_inductive.md / .json
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from src.features import FEATURE_NAMES, account_features
from src.models.graphsage import NodeRiskModel

ROOT       = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
MODEL_PATH = os.path.join(ROOT, "models", "graphsage_inductive.pt")
REPORTS    = os.path.join(ROOT, "reports")

WINDOWS = {                       # name -> (first row, last row)
    "train_1": (0, 1_060_000), "train_2": (1_060_000, 2_120_000),
    "val": (2_120_000, 3_180_000), "test": (3_180_000, 4_240_000), "late": (5_300_000, 6_362_620),
}
HIDDEN, EMBED, DROPOUT = 64, 64, 0.3
MAX_POS_WEIGHT = 50.0             # the raw class ratio is in the hundreds; capping keeps training stable
SEEDS = [0, 1, 2]


def log(*args):
    print(*args, flush=True)


def build_graph(df: pd.DataFrame) -> dict:
    """One time window -> features, edges and labels. Labels never touch the features."""
    codes, accounts = pd.factorize(pd.concat([df["nameOrig"], df["nameDest"]], ignore_index=True))
    n = len(df)
    src, dst = codes[:n], codes[n:]
    x = account_features(src, dst, df["amount"].to_numpy(), len(accounts))
    y = np.zeros(len(accounts), dtype=np.float32)
    y[dst[df["isFraud"].to_numpy() == 1]] = 1.0             # account received fraud-labelled money
    receivers = np.bincount(dst, minlength=len(accounts)) > 0
    edge_index = torch.from_numpy(np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])).long()
    return {"x": torch.from_numpy(x), "edge_index": edge_index, "y": torch.from_numpy(y),
            "receivers": receivers, "n_tx": n}


def precision_at(y, score, k):
    return float(y[np.argsort(-score)[:k]].mean())


def metrics(y, score) -> dict:
    return {"pr_auc": float(average_precision_score(y, score)), "roc_auc": float(roc_auc_score(y, score)),
            "precision_at_100": precision_at(y, score, 100), "precision_at_1000": precision_at(y, score, 1000),
            "base_rate": float(y.mean())}


@torch.no_grad()
def predict(model, g):
    model.eval()
    return torch.sigmoid(model(g["x"], g["edge_index"])).numpy()


def evaluate(graphs, score_fn) -> dict:
    """Score every held-out window. Only receiving accounts are scored: a pure sender cannot be a mule."""
    out = {}
    for name in ("val", "test", "late"):
        g = graphs[name]
        r = g["receivers"]
        out[name] = metrics(g["y"].numpy()[r], score_fn(g)[r])
    return out


def train_gnn(graphs, seed, epochs=80, patience=12):
    torch.manual_seed(seed)
    model = NodeRiskModel(len(FEATURE_NAMES), HIDDEN, EMBED, DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)
    train = [graphs["train_1"], graphs["train_2"]]
    pos = sum(float(g["y"][torch.from_numpy(g["receivers"])].sum()) for g in train)
    neg = sum(float(g["receivers"].sum()) for g in train) - pos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(neg / pos, MAX_POS_WEIGHT)))
    val = graphs["val"]
    y_val = val["y"].numpy()[val["receivers"]]

    best, best_state, wait = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for g in train:
            mask = torch.from_numpy(g["receivers"])
            opt.zero_grad()
            loss = loss_fn(model(g["x"], g["edge_index"])[mask], g["y"][mask])
            loss.backward()
            opt.step()
            total += loss.item()
        ap = average_precision_score(y_val, predict(model, val)[val["receivers"]])
        improved = ap > best
        if improved:
            best, wait = ap, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        log(f"  seed {seed} epoch {epoch:>3} loss {total / len(train):.4f} val PR-AUC {ap:.4f} "
            f"{'*' if improved else ' '} ({time.time() - t0:.0f}s)")
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    return model, best


def write_report(summary, runtime):
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, "paysim_inductive.json"), "w") as f:
        json.dump({"windows": WINDOWS, "features": FEATURE_NAMES, "seeds": SEEDS, "results": summary,
                   "runtime_seconds": runtime}, f, indent=2)

    lines = ["# PaySim — inductive account-risk model, without label leakage", "",
             "Task: predict which accounts receive fraud-labelled money. Features are structure and amounts only "
             "(`src/features.py`), standardised per graph. Each time window is a separate graph, so the test "
             "graphs are entirely unseen during training. Scores are over receiving accounts; "
             f"models are averaged over {len(SEEDS)} seeds.", ""]
    titles = {"test": "Test window (steps 239-306)",
              "late": "Late window (steps 373-743), where fraud is far more common"}
    for split, title in titles.items():
        base = summary["XGBoost (no graph)"][split]["base_rate"]["mean"]
        lines += [f"## {title}", "", f"Base rate: {base:.3%} of receiving accounts.", "",
                  "| Model | PR-AUC | ROC-AUC | Precision@100 | Precision@1000 |", "|---|---|---|---|---|"]
        for model_name, s in summary.items():
            r = s[split]
            lines.append(f"| {model_name} | {r['pr_auc']['mean']:.3f} ± {r['pr_auc']['std']:.3f} | "
                         f"{r['roc_auc']['mean']:.3f} | {r['precision_at_100']['mean']:.2f} | "
                         f"{r['precision_at_1000']['mean']:.2f} |")
        lines.append("")
    path = os.path.join(REPORTS, "paysim_inductive.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paysim", default=os.path.join(ROOT, "data", "raw", "paysim", "paysim.csv"))
    args = parser.parse_args()

    t_start = time.time()
    log("Reading PaySim...")
    df = pd.read_csv(args.paysim, usecols=["step", "amount", "nameOrig", "nameDest", "isFraud"])
    graphs = {}
    for name, (a, b) in WINDOWS.items():
        graphs[name] = g = build_graph(df.iloc[a:b])
        r = g["receivers"]
        log(f"  {name:<8} {g['n_tx']:>9,} tx | {len(g['y']):>9,} accounts | {int(r.sum()):>8,} receivers | "
            f"{int(g['y'].sum()):>5,} fraud receivers ({g['y'].numpy()[r].mean():.3%})")
    del df

    results = {}

    # Single-feature heuristic: the largest amount the account ever received
    col = FEATURE_NAMES.index("max_received")
    results["Largest receipt (1 feature)"] = [evaluate(graphs, lambda g: g["x"].numpy()[:, col])]

    log("\nXGBoost (same features, no graph)...")
    train_names = ("train_1", "train_2")
    xtr = np.vstack([graphs[n]["x"].numpy()[graphs[n]["receivers"]] for n in train_names])
    ytr = np.concatenate([graphs[n]["y"].numpy()[graphs[n]["receivers"]] for n in train_names])
    runs = []
    for seed in SEEDS:
        xgb = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                            scale_pos_weight=min((ytr == 0).sum() / ytr.sum(), MAX_POS_WEIGHT),
                            eval_metric="aucpr", n_jobs=-1, random_state=seed)
        xgb.fit(xtr, ytr)
        runs.append(evaluate(graphs, lambda g, m=xgb: m.predict_proba(g["x"].numpy())[:, 1]))
        log(f"  seed {seed} test PR-AUC {runs[-1]['test']['pr_auc']:.4f}")
    results["XGBoost (no graph)"] = runs

    log("\nGraphSAGE (inductive)...")
    runs, best_model, best_val = [], None, -1.0
    for seed in SEEDS:
        model, val_ap = train_gnn(graphs, seed)
        runs.append(evaluate(graphs, lambda g, m=model: predict(m, g)))
        log(f"  seed {seed} test PR-AUC {runs[-1]['test']['pr_auc']:.4f}")
        if val_ap > best_val:
            best_val, best_model = val_ap, model
    results["GraphSAGE (inductive)"] = runs

    summary = {
        model_name: {
            split: {k: {"mean": float(np.mean([r[split][k] for r in runs])),
                        "std": float(np.std([r[split][k] for r in runs]))} for k in runs[0][split]}
            for split in ("val", "test", "late")
        } for model_name, runs in results.items()
    }

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save({"state_dict": best_model.state_dict(), "features": FEATURE_NAMES,
                "hidden": HIDDEN, "embedding": EMBED, "dropout": DROPOUT,
                "trained_on": "PaySim rows 0-2.12M (steps 1-183)", "val_pr_auc": best_val}, MODEL_PATH)

    path = write_report(summary, round(time.time() - t_start))
    log(f"\nDone in {time.time() - t_start:.0f}s. Model saved to {MODEL_PATH}\n")
    log(open(path, encoding="utf-8").read())


if __name__ == "__main__":
    main()
