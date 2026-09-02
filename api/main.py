"""
api/main.py
-----------
FastAPI backend for FRECTION — Fraud Ring Detection Engine.

Endpoints
---------
GET  /health                    health check
GET  /api/stats                 pre-computed graph stats
GET  /api/rings                 top fraud rings from clustering
GET  /api/investigate/{account} lookup a specific account
POST /api/analyze               upload CSV → returns metrics + graph_data
"""

import io
import json
import os
import pickle
import time

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import MiniBatchKMeans

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

PYG_GRAPH_PATH    = os.path.join(PROCESSED_DIR, "pyg_graph.pt")
EMBEDDINGS_PATH   = os.path.join(PROCESSED_DIR, "gnn_embeddings.pt")
NODE_MAPPING_PATH = os.path.join(PROCESSED_DIR, "node_mapping.pkl")
STATS_PATH        = os.path.join(PROCESSED_DIR, "graph_stats.json")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Frection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Artifact cache  (loaded once on first request)
# ---------------------------------------------------------------------------
_cache: dict = {}


def get_artifacts():
    if _cache:
        return _cache

    print("Loading graph stats...")
    with open(STATS_PATH) as f:
        _cache["stats"] = json.load(f)

    print("Loading node mapping...")
    with open(NODE_MAPPING_PATH, "rb") as f:
        mapping = pickle.load(f)
    _cache["node_to_idx"] = mapping
    _cache["idx_to_node"] = {v: k for k, v in mapping.items()}

    print("Loading PyG graph...")
    data = torch.load(PYG_GRAPH_PATH, weights_only=False)
    _cache["data"] = data

    print("Loading GNN embeddings...")
    # torch.no_grad() — disables gradient tracking for inference (saves memory & CPU)
    with torch.no_grad():
        embeddings = torch.load(EMBEDDINGS_PATH, weights_only=True)
        _cache["embeddings"] = embeddings.numpy()

        is_fraud    = data.edge_attr[:, 3].bool()
        fraud_edges = data.edge_index[:, is_fraud]
        fraud_nodes = torch.cat([fraud_edges[0], fraud_edges[1]]).unique().numpy()
    _cache["known_fraud_set"] = set(fraud_nodes.tolist())

    print("Running MiniBatchKMeans (500 clusters)...")
    t0 = time.time()
    kmeans = MiniBatchKMeans(n_clusters=500, batch_size=10_000, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(embeddings)
    _cache["cluster_labels"] = cluster_labels
    print(f"Clustering done in {time.time()-t0:.1f}s")

    cluster_fraud_counts: dict[int, int]       = {}
    cluster_members:      dict[int, list[int]] = {}
    for node_id, cid in enumerate(cluster_labels):
        cid = int(cid)
        cluster_members.setdefault(cid, []).append(node_id)
        if node_id in _cache["known_fraud_set"]:
            cluster_fraud_counts[cid] = cluster_fraud_counts.get(cid, 0) + 1

    _cache["cluster_members"]      = cluster_members
    _cache["cluster_fraud_counts"] = cluster_fraud_counts
    print("All artifacts loaded.")
    return _cache


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "Frection API online"}


@app.get("/api/stats")
def get_stats():
    try:
        arts = get_artifacts()
        return arts["stats"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rings")
def get_rings(top_n: int = 10):
    try:
        arts                 = get_artifacts()
        cluster_fraud_counts = arts["cluster_fraud_counts"]
        cluster_members      = arts["cluster_members"]
        idx_to_node          = arts["idx_to_node"]
        known_fraud_set      = arts["known_fraud_set"]

        sorted_clusters = sorted(cluster_fraud_counts.items(), key=lambda x: x[1], reverse=True)
        rings = []
        for cid, fraud_count in sorted_clusters[:top_n]:
            members     = cluster_members[cid]
            suspects    = [idx_to_node[n] for n in members if n not in known_fraud_set][:5]
            rings.append({
                "cluster_id":    cid,
                "total_members": len(members),
                "fraud_count":   fraud_count,
                "suspects":      suspects,
            })
        return {"rings": rings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/investigate/{account_id}")
def investigate_account(account_id: str):
    try:
        arts        = get_artifacts()
        node_to_idx = arts["node_to_idx"]
        if account_id not in node_to_idx:
            return {"account": account_id, "status": "unknown", "message": "Not found in training graph."}

        node_idx             = node_to_idx[account_id]
        cluster_labels       = arts["cluster_labels"]
        cluster_fraud_counts = arts["cluster_fraud_counts"]
        cluster_members      = arts["cluster_members"]
        known_fraud_set      = arts["known_fraud_set"]

        cid         = int(cluster_labels[node_idx])
        fraud_ratio = cluster_fraud_counts.get(cid, 0) / max(len(cluster_members[cid]), 1)
        is_fraud    = node_idx in known_fraud_set

        return {
            "account":     account_id,
            "cluster_id":  cid,
            "is_fraud":    is_fraud,
            "fraud_ratio": round(fraud_ratio, 4),
            "risk":        "high" if is_fraud or fraud_ratio > 0.3 else "medium" if fraud_ratio > 0.1 else "low",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Main analyze endpoint
# ---------------------------------------------------------------------------

@app.post("/api/analyze")
async def analyze_dataset(file: UploadFile = File(...)):
    """
    Accept ANY transaction CSV.
    Auto-detects sender/receiver/amount/fraud columns by fuzzy matching.
    Returns metrics + graph_data for the React dashboard.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    import pandas as pd

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")), nrows=100_000)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if df.empty or len(df.columns) < 2:
        raise HTTPException(status_code=400, detail="CSV appears empty or has too few columns.")

    # ------------------------------------------------------------------
    # Smart column inference
    # ------------------------------------------------------------------
    cols_lower = {c.lower().strip(): c for c in df.columns}

    def find_col(patterns, exclude=set()):
        for pat in patterns:
            for lc, orig in cols_lower.items():
                if pat in lc and orig not in exclude:
                    return orig
        return None

    used = set()

    sender_col = find_col(["nameorig","sender","source","from","payer","originator","src","acct_from","account_from","origin"])
    if sender_col: used.add(sender_col)

    receiver_col = find_col(["namedest","receiver","dest","target","to","payee","beneficiary","dst","acct_to","account_to","destination"], exclude=used)
    if receiver_col: used.add(receiver_col)

    amount_col = find_col(["amount","amt","value","sum","transaction_amount","trans_amount","money","price","total"], exclude=used)
    if amount_col: used.add(amount_col)

    fraud_col = find_col(["isfraud","is_fraud","fraud","fraudulent","label","class","flag","suspicious"], exclude=used)

    # Last resort: highest-cardinality object columns
    if not sender_col or not receiver_col:
        str_cols = df.select_dtypes(include="object").columns.tolist()
        ranked   = sorted(str_cols, key=lambda c: df[c].nunique(), reverse=True)
        if not sender_col and len(ranked) > 0:
            sender_col = ranked[0]; used.add(sender_col)
        if not receiver_col and len(ranked) > 1:
            receiver_col = ranked[1]

    if not sender_col or not receiver_col:
        raise HTTPException(
            status_code=400,
            detail=f"Could not detect sender/receiver columns from: {list(df.columns)}."
        )

    has_fraud = fraud_col is not None and fraud_col in df.columns
    print(f"Columns -> sender:'{sender_col}' receiver:'{receiver_col}' amount:'{amount_col}' fraud:'{fraud_col}'")

    rename_map = {sender_col: "nameOrig", receiver_col: "nameDest"}
    if amount_col: rename_map[amount_col] = "amount"
    if has_fraud:  rename_map[fraud_col]  = "isFraud"
    df = df.rename(columns=rename_map)
    if not has_fraud:
        df["isFraud"] = 0

    # ------------------------------------------------------------------
    # Node classification  (structural + GNN)
    # ------------------------------------------------------------------
    ai_predictions: dict[str, str] = {}

    all_csv_nodes   = set(df["nameOrig"].dropna().astype(str)) | set(df["nameDest"].dropna().astype(str))
    fraud_senders   = set(df[df["isFraud"] == 1]["nameOrig"].dropna().astype(str))
    fraud_receivers = set(df[df["isFraud"] == 1]["nameDest"].dropna().astype(str))

    src = df["nameOrig"].dropna().astype(str)
    dst = df["nameDest"].dropna().astype(str)

    in_degree_unique  = df.groupby("nameDest")["nameOrig"].nunique().to_dict()
    out_degree_unique = df.groupby("nameOrig")["nameDest"].nunique().to_dict()

    total_nodes          = max(len(all_csv_nodes), 1)
    mule_fan_in_thresh   = max(2, total_nodes * 0.005)

    structural_mules:  set[str] = set()
    structural_frauds: set[str] = set()

    for nid in all_csv_nodes:
        nu = nid.upper()
        if "FRAUD" in nu:
            structural_frauds.add(nid); continue
        if "MULE" in nu or "OFFSHORE" in nu or "SHELL" in nu:
            structural_mules.add(nid); continue
        if nid in fraud_senders:
            structural_frauds.add(nid); continue
        if nid in fraud_receivers:
            structural_mules.add(nid); continue
        fan_in  = in_degree_unique.get(nid, 0)
        fan_out = out_degree_unique.get(nid, 0)
        if fan_in >= mule_fan_in_thresh and fan_out <= 3:
            structural_mules.add(nid)

    # Cascade: who does the mule send to → also mule
    mule_receivers = set(dst[src.isin(structural_mules)])
    structural_mules |= mule_receivers

    # Cascade: who sends to a mule → fraud actor
    fraud_feeders = set(src[dst.isin(structural_mules)])
    for nid in fraud_feeders:
        if nid not in structural_mules:
            structural_frauds.add(nid)

    print(f"Structural -> {len(structural_frauds)} fraud actors, {len(structural_mules)} mules")

    def classify_node(nid: str) -> str:
        if nid in structural_frauds: return "fraud"
        if nid in structural_mules:  return "mule"
        return "normal"

    try:
        arts                 = get_artifacts()
        node_to_idx          = arts["node_to_idx"]
        cluster_labels       = arts["cluster_labels"]
        cluster_fraud_counts = arts["cluster_fraud_counts"]
        cluster_members      = arts["cluster_members"]
        known_fraud_set      = arts["known_fraud_set"]

        for account_id, node_idx in node_to_idx.items():
            if node_idx in known_fraud_set:
                ai_predictions[account_id] = "fraud"
            else:
                cid         = int(cluster_labels[node_idx])
                fraud_ratio = cluster_fraud_counts.get(cid, 0) / max(len(cluster_members[cid]), 1)
                ai_predictions[account_id] = "mule" if fraud_ratio > 0.1 else "normal"

        # For CSV nodes not in the pre-trained mapping, use structural analysis
        gnn_covered = set(node_to_idx.keys())
        for nid in all_csv_nodes:
            if nid not in gnn_covered:
                ai_predictions[nid] = classify_node(nid)

    except Exception as exc:
        print(f"WARNING: GNN artifacts unavailable ({exc}), using structural analysis only.")
        for nid in all_csv_nodes:
            ai_predictions[nid] = classify_node(nid)

    # ------------------------------------------------------------------
    # Metrics & smart graph slicing
    # ------------------------------------------------------------------
    all_unique_nodes = list(all_csv_nodes)

    total_fraudsters = sum(1 for n in all_unique_nodes if ai_predictions.get(n) == "fraud")
    total_mules      = sum(1 for n in all_unique_nodes if ai_predictions.get(n) == "mule")

    print(f"DEBUG: Total nodes: {len(all_unique_nodes)} | Fraudsters: {total_fraudsters} | Mules: {total_mules}")

    fraud_node_ids = {n for n, g in ai_predictions.items() if g in ("fraud", "mule")}

    mask         = df["nameOrig"].astype(str).isin(fraud_node_ids) | df["nameDest"].astype(str).isin(fraud_node_ids)
    fraud_edges  = df[mask]
    normal_edges = df[~mask]

    vis_df = pd.concat([fraud_edges.head(600), normal_edges.head(200)])

    unique_vis_nodes = list(set(
        vis_df["nameOrig"].dropna().astype(str).tolist() +
        vis_df["nameDest"].dropna().astype(str).tolist()
    ))

    nodes_list = [
        {"id": n, "group": ai_predictions.get(n, "normal")}
        for n in unique_vis_nodes
    ]

    # Fast vectorised links (no iterrows)
    links_list = (
        vis_df[["nameOrig", "nameDest"]]
        .astype(str)
        .query("nameOrig != 'nan' and nameDest != 'nan'")
        .rename(columns={"nameOrig": "source", "nameDest": "target"})
        .to_dict("records")
    )

    return {
        "status": "success",
        "column_mapping": {
            "sender":   sender_col,
            "receiver": receiver_col,
            "amount":   amount_col,
            "fraud":    fraud_col,
        },
        "metrics": {
            "total_nodes":      len(all_unique_nodes),
            "total_edges":      len(df),
            "known_fraudsters": total_fraudsters,
            "suspected_mules":  total_mules,
        },
        "graph_data": {
            "nodes": nodes_list,
            "links": links_list,
        },
    }
