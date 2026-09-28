"""
api/main.py
-----------
FastAPI backend for FRECTION — Fraud Ring Detection Engine.

Endpoints
---------
GET  /health                    health check (+ model warm-up status)
GET  /api/stats                 pre-computed graph stats
GET  /api/rings                 top fraud rings from clustering
GET  /api/investigate/{account} lookup a specific account
POST /api/analyze               upload CSV → returns metrics + graph_data

Performance notes
-----------------
* Heavy artifacts (torch, PyG graph, embeddings, KMeans) are loaded in a
  background thread as soon as the server starts, so the first upload no
  longer pays the warm-up cost.
* KMeans cluster labels are cached to disk and reused on later restarts
  (invalidated automatically if the embeddings file changes).
* Per-account GNN labels are pre-computed once, so each upload only looks up
  the accounts that are actually in the CSV.
* Node classification is fully vectorised with pandas / numpy.
"""

import json
import os
import pickle
import sys
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api import explain as explainer  # noqa: E402
from api.entities import analyze_entities  # noqa: E402
from api.ingest import (  # noqa: E402
    binary_label, detect_columns, detect_id_column, is_text, normalise_frame, parse_numeric_text, read_upload,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

PYG_GRAPH_PATH     = os.path.join(PROCESSED_DIR, "pyg_graph.pt")
EMBEDDINGS_PATH    = os.path.join(PROCESSED_DIR, "gnn_embeddings.pt")
NODE_MAPPING_PATH  = os.path.join(PROCESSED_DIR, "node_mapping.pkl")
STATS_PATH         = os.path.join(PROCESSED_DIR, "graph_stats.json")
CLUSTER_CACHE_PATH = os.path.join(PROCESSED_DIR, "cluster_labels_cache.npz")

N_CLUSTERS        = 500
MAX_ROWS          = 100_000
MULE_RATIO_THRESH = 0.1
NONE_COLUMN       = "__none__"   # sent by the UI when the user picks "no column"

# ---------------------------------------------------------------------------
# Artifact cache  (loaded once, in the background, at startup)
# ---------------------------------------------------------------------------
_cache: dict = {}
_load_lock   = threading.Lock()
_load_done   = threading.Event()
_load_error: str | None = None


def _cluster_labels(embeddings: np.ndarray) -> np.ndarray:
    """MiniBatchKMeans labels, cached on disk keyed by the embeddings file."""
    st  = os.stat(EMBEDDINGS_PATH)
    key = np.array([st.st_size, st.st_mtime_ns, N_CLUSTERS, len(embeddings)], dtype=np.int64)

    if os.path.exists(CLUSTER_CACHE_PATH):
        try:
            cached = np.load(CLUSTER_CACHE_PATH)
            if np.array_equal(cached["key"], key):
                print("Loaded cached cluster labels.")
                return cached["labels"]
        except Exception as exc:
            print(f"Ignoring unreadable cluster cache ({exc}).")

    from sklearn.cluster import MiniBatchKMeans

    print(f"Running MiniBatchKMeans ({N_CLUSTERS} clusters)...")
    t0 = time.time()
    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, batch_size=10_000, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    print(f"Clustering done in {time.time()-t0:.1f}s")

    try:
        np.savez(CLUSTER_CACHE_PATH, key=key, labels=labels)
    except OSError as exc:
        print(f"Could not write cluster cache ({exc}).")
    return labels


def _load_artifacts() -> dict:
    import torch  # imported lazily — it is slow to import and only needed here

    arts: dict = {}

    print("Loading graph stats...")
    with open(STATS_PATH) as f:
        arts["stats"] = json.load(f)

    print("Loading node mapping...")
    with open(NODE_MAPPING_PATH, "rb") as f:
        mapping = pickle.load(f)
    arts["node_to_idx"] = mapping
    arts["idx_to_node"] = {v: k for k, v in mapping.items()}

    print("Loading PyG graph...")
    data = torch.load(PYG_GRAPH_PATH, weights_only=False)

    print("Loading GNN embeddings...")
    with torch.no_grad():
        embeddings  = torch.load(EMBEDDINGS_PATH, weights_only=True).numpy()
        is_fraud    = data.edge_attr[:, 3].bool()
        fraud_edges = data.edge_index[:, is_fraud]
        fraud_nodes = torch.cat([fraud_edges[0], fraud_edges[1]]).unique().numpy()
    arts["embeddings"]      = embeddings
    arts["known_fraud_set"] = set(fraud_nodes.tolist())

    cluster_labels = _cluster_labels(embeddings).astype(np.int64)
    arts["cluster_labels"] = cluster_labels

    n_nodes     = len(cluster_labels)
    known_fraud = np.zeros(n_nodes, dtype=bool)
    known_fraud[fraud_nodes[fraud_nodes < n_nodes]] = True

    sizes        = np.bincount(cluster_labels)
    fraud_counts = np.bincount(cluster_labels[known_fraud], minlength=len(sizes))

    order           = np.argsort(cluster_labels, kind="stable")
    boundaries      = np.cumsum(sizes)[:-1]
    cluster_members = {cid: m.tolist() for cid, m in enumerate(np.split(order, boundaries)) if len(m)}

    arts["cluster_members"]      = cluster_members
    arts["cluster_fraud_counts"] = {int(c): int(n) for c, n in enumerate(fraud_counts) if n}

    # Pre-compute the GNN verdict for every known account once, instead of on every upload.
    ratio      = fraud_counts / np.maximum(sizes, 1)
    node_ratio = ratio[cluster_labels]
    node_group = np.where(known_fraud, "fraud", np.where(node_ratio > MULE_RATIO_THRESH, "mule", "normal"))

    idx_positions      = np.fromiter(mapping.values(), dtype=np.int64, count=len(mapping))
    arts["gnn_index"]  = pd.Index(list(mapping.keys()))
    arts["gnn_groups"] = node_group[idx_positions]

    print("All artifacts loaded.")
    return arts


def _warm_up():
    global _load_error
    t0 = time.time()
    try:
        with _load_lock:
            if not _cache:
                _cache.update(_load_artifacts())
        print(f"Model warm-up finished in {time.time()-t0:.1f}s")
    except Exception as exc:
        _load_error = str(exc)
        print(f"WARNING: GNN artifacts unavailable ({exc}); structural analysis only.")
    finally:
        _load_done.set()


def get_artifacts() -> dict:
    """Return the loaded artifacts, waiting for the warm-up if it is still running."""
    _load_done.wait()
    if _load_error:
        raise RuntimeError(_load_error)
    return _cache


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Frection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _model_status() -> str:
    if not _load_done.is_set():
        return "loading"
    return "unavailable" if _load_error else "ready"


@app.get("/health")
@app.get("/api/health")
def health_check():
    return {"status": "Frection API online", "model": _model_status(), "ai_explanations": explainer.ai_available()}


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

def _as_str_ids(col: pd.Series) -> pd.Series:
    """Account IDs as strings, keeping missing values as NaN."""
    # Numeric IDs with blanks are parsed as float ("2.0"); restore them to "2".
    if pd.api.types.is_float_dtype(col):
        valid = col.dropna()
        if (valid == np.floor(valid)).all():
            col = col.astype("Int64")
    return col.astype(str).where(col.notna())


def _analyze_transactions(df: pd.DataFrame, sender_col, receiver_col, fraud_col, amount_col=None):
    """Graph analysis for sender → receiver transaction ledgers.

    Also returns the per-account evidence the explainer uses to say *why*
    an account was labelled fraud / mule.
    """
    has_fraud = fraud_col is not None and fraud_col in df.columns

    # ------------------------------------------------------------------
    # Node classification  (structural + GNN), vectorised
    # ------------------------------------------------------------------
    s_full = _as_str_ids(df[sender_col])     # aligned with df, NaN where missing
    d_full = _as_str_ids(df[receiver_col])
    if has_fraud:
        fraud_row = binary_label(df[fraud_col])
    else:
        fraud_row = np.zeros(len(df), dtype=bool)

    # Map every account to an integer code once (-1 = missing); all graph logic below
    # works on these codes instead of hashing strings over and over.
    n_rows       = len(df)
    codes, nodes = pd.factorize(pd.concat([s_full, d_full], ignore_index=True))
    nodes        = pd.Index(nodes, dtype=object)
    n_nodes      = len(nodes)
    if n_nodes == 0:
        raise HTTPException(status_code=400, detail=f"No account IDs found in '{sender_col}' / '{receiver_col}'.")
    sc, dc       = codes[:n_rows], codes[n_rows:]
    s_ok, d_ok   = sc >= 0, dc >= 0
    both_ok      = s_ok & d_ok

    def flag(idx):
        out = np.zeros(n_nodes, dtype=bool)
        out[idx] = True
        return out

    is_fs = flag(sc[fraud_row & s_ok])
    is_fr = flag(dc[fraud_row & d_ok])

    # Unique in/out neighbours per account
    pair_ids = np.unique(sc[both_ok].astype(np.int64) * n_nodes + dc[both_ok])
    fan_in   = np.bincount(pair_ids % n_nodes, minlength=n_nodes)
    fan_out  = np.bincount(pair_ids // n_nodes, minlength=n_nodes)

    total_nodes        = max(n_nodes, 1)
    mule_fan_in_thresh = max(2, total_nodes * 0.005)
    fan_rule           = (fan_in >= mule_fan_in_thresh) & (fan_out <= 3)

    upper     = nodes.str.upper()
    name_frd  = np.asarray(upper.str.contains("FRAUD", regex=False), dtype=bool)
    name_mule = np.asarray(upper.str.contains("MULE|OFFSHORE|SHELL", regex=True), dtype=bool)

    # Priority: FRAUD in name > MULE/OFFSHORE/SHELL in name > fraud sender > fraud receiver > fan-in rule
    struct_fraud = name_frd | (~name_mule & is_fs)
    struct_mule  = ~name_frd & (name_mule | (~is_fs & (is_fr | fan_rule)))

    base_fraud, base_mule = struct_fraud.copy(), struct_mule.copy()

    # Cascade: who does the mule send to → also mule
    struct_mule |= flag(dc[both_ok & struct_mule[sc]])

    # Cascade: who sends to a mule → fraud actor
    struct_fraud |= flag(sc[both_ok & struct_mule[dc]]) & ~struct_mule

    print(f"Structural -> {int(struct_fraud.sum())} fraud actors, {int(struct_mule.sum())} mules")

    groups = np.where(struct_fraud, "fraud", np.where(struct_mule, "mule", "normal")).astype(object)

    engine  = "structural"
    gnn_pos = np.full(n_nodes, -1)
    try:
        arts = get_artifacts()
        pos  = arts["gnn_index"].get_indexer(nodes)
        gnn_pos = pos
        hit  = pos >= 0
        groups[hit] = arts["gnn_groups"][pos[hit]]
        engine = "gnn+structural"
    except Exception as exc:
        print(f"WARNING: GNN artifacts unavailable ({exc}), using structural analysis only.")

    # ------------------------------------------------------------------
    # Metrics & smart graph slicing
    # ------------------------------------------------------------------
    total_fraudsters = int((groups == "fraud").sum())
    total_mules      = int((groups == "mule").sum())
    print(f"Total nodes: {len(nodes)} | Fraudsters: {total_fraudsters} | Mules: {total_mules}")

    flagged = groups != "normal"
    mask    = (s_ok & flagged[sc]) | (d_ok & flagged[dc])

    vis_idx = np.concatenate([np.flatnonzero(mask)[:600], np.flatnonzero(~mask)[:200]])
    vis_s   = s_full.iloc[vis_idx]
    vis_d   = d_full.iloc[vis_idx]

    group_of   = pd.Series(groups, index=nodes)
    vis_nodes  = pd.unique(pd.concat([vis_s, vis_d], ignore_index=True).dropna())
    vis_groups = group_of.reindex(vis_nodes).fillna("normal").tolist()
    nodes_list = [{"id": n, "group": g} for n, g in zip(vis_nodes.tolist(), vis_groups)]

    both       = (vis_s.notna() & vis_d.notna()).to_numpy()
    links_list = [{"source": s, "target": t} for s, t in zip(vis_s[both].tolist(), vis_d[both].tolist())]

    metrics = {
        "total_nodes":      len(nodes),
        "total_edges":      len(df),
        "known_fraudsters": total_fraudsters,
        "suspected_mules":  total_mules,
    }
    if amount_col and amount_col in df.columns:
        amt = df[amount_col]
        amounts = (parse_numeric_text(amt) if is_text(amt) else pd.to_numeric(amt, errors="coerce")).to_numpy(dtype=float)
    else:
        amounts = None

    evidence = {
        "nodes": nodes, "groups": groups, "sc": sc, "dc": dc, "fraud_row": fraud_row,
        "has_label": has_fraud, "amounts": amounts,
        "name_frd": name_frd, "name_mule": name_mule, "is_fs": is_fs, "is_fr": is_fr,
        "fan_in": fan_in, "fan_out": fan_out, "fan_thresh": mule_fan_in_thresh, "fan_rule": fan_rule,
        "base_fraud": base_fraud, "base_mule": base_mule,
        "struct_fraud": struct_fraud, "struct_mule": struct_mule, "gnn_pos": gnn_pos,
    }
    return metrics, {"nodes": nodes_list, "links": links_list}, engine, evidence


def _pick(df: pd.DataFrame, value, role: str):
    """Validate a user-chosen column. '' → keep auto-detection, '__none__' → no column."""
    if value is None or value == "":
        return ...
    if value == NONE_COLUMN:
        return None
    if value not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{value}' chosen as {role} is not in the file.")
    return value


# Plain (sync) handler: FastAPI runs it in a worker thread, so the CPU-heavy
# analysis no longer blocks the event loop and other requests.
@app.post("/api/analyze")
def analyze_dataset(
    file: UploadFile = File(...),
    mode: str = Form("auto"),                 # auto | transactions | entities
    sender_col: str | None = Form(None),
    receiver_col: str | None = Form(None),
    amount_col: str | None = Form(None),
    label_col: str | None = Form(None),
    id_col: str | None = Form(None),
):
    """
    Accept ANY CSV.
    * Transaction ledgers (sender → receiver) get the graph + GNN analysis.
    * One-row-per-customer tables get anomaly detection + a similarity graph.
    Columns are detected automatically; the optional form fields override them.
    Returns metrics + graph_data for the React dashboard.
    """
    t_start = time.perf_counter()

    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    contents = file.file.read()
    if not contents.strip():
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    try:
        df = read_upload(contents, MAX_ROWS)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    rows_truncated = len(df) > MAX_ROWS
    if rows_truncated:
        df = df.iloc[:MAX_ROWS]
    df = normalise_frame(df)

    if df.empty or len(df.columns) < 2:
        raise HTTPException(status_code=400, detail="CSV appears empty or has too few columns (need at least 2).")

    # ------------------------------------------------------------------
    # Understand the columns (auto-detect, then apply any user overrides)
    # ------------------------------------------------------------------
    mapping = detect_columns(df)
    picked  = {
        "sender":   _pick(df, sender_col, "sender"),
        "receiver": _pick(df, receiver_col, "receiver"),
        "amount":   _pick(df, amount_col, "amount"),
        "fraud":    _pick(df, label_col, "label"),
        "id":       _pick(df, id_col, "customer ID"),
    }
    if mode == "transactions" or picked["sender"] not in (..., None) or picked["receiver"] not in (..., None):
        mapping["mode"] = "transactions"
    elif mode == "entities":
        if mapping["mode"] != "entities":
            mapping.update(mode="entities", sender=None, receiver=None, id=detect_id_column(df, {mapping["fraud"]}))
    for role, value in picked.items():
        if value is not ...:
            mapping[role] = value

    print(f"Columns -> {mapping}")

    feature_names: list[str] = []
    try:
        if mapping["mode"] == "transactions":
            if not mapping["sender"] or not mapping["receiver"]:
                raise HTTPException(status_code=400, detail="Pick both a sender and a receiver column for transaction mode.")
            if mapping["sender"] == mapping["receiver"]:
                raise HTTPException(status_code=400, detail="Sender and receiver must be different columns.")
            metrics, graph_data, engine, evidence = _analyze_transactions(
                df, mapping["sender"], mapping["receiver"], mapping["fraud"], mapping["amount"])
        else:
            metrics, graph_data, engine, feature_names, evidence = analyze_entities(
                df, mapping["id"], mapping["fraud"], mapping["amount"], _as_str_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    analysis_id = explainer.store_analysis(mapping["mode"], mapping, evidence)

    elapsed_ms = round((time.perf_counter() - t_start) * 1000)
    print(f"Analysis finished in {elapsed_ms} ms ({engine})")

    return {
        "status": "success",
        "analysis_id": analysis_id,
        "ai_explanations": explainer.ai_available(),
        "mode":   mapping["mode"],
        "columns": [str(c) for c in df.columns],
        "column_mapping": {
            "sender":   mapping["sender"],
            "receiver": mapping["receiver"],
            "amount":   mapping["amount"],
            "fraud":    mapping["fraud"],
            "id":       mapping["id"],
        },
        "metrics":    metrics,
        "graph_data": graph_data,
        "meta": {
            "elapsed_ms":     elapsed_ms,
            "engine":         engine,
            "rows_truncated": rows_truncated,
            "max_rows":       MAX_ROWS,
            "features_used":  feature_names[:40],
            "features_count": len(feature_names),
        },
    }


# ---------------------------------------------------------------------------
# Account investigator: search accounts, explain a verdict, optional AI write-up
# ---------------------------------------------------------------------------

def _analysis_or_404(analysis_id: str) -> dict:
    entry = explainer.get_analysis(analysis_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="This analysis has expired — please run the detection again.")
    return entry


def _artifacts_if_ready():
    return _cache if _load_done.is_set() and not _load_error else None


@app.get("/api/analysis/{analysis_id}/accounts")
def search_accounts(analysis_id: str, q: str = "", limit: int = 50, group: str = ""):
    return explainer.list_accounts(_analysis_or_404(analysis_id), q, max(1, min(limit, 200)), group)


@app.get("/api/analysis/{analysis_id}/explain")
def explain_account(analysis_id: str, account: str):
    result = explainer.explain(_analysis_or_404(analysis_id), account, _artifacts_if_ready())
    if result is None:
        raise HTTPException(status_code=404, detail=f"Account '{account}' is not in this dataset.")
    return result


@app.post("/api/analysis/{analysis_id}/ai")
def ai_explain_account(analysis_id: str, account: str = Body(...), question: str | None = Body(None)):
    if not explainer.ai_available():
        raise HTTPException(status_code=400, detail="AI explanations are off. Add ANTHROPIC_API_KEY to a .env file in the project root and restart the backend.")
    result = explainer.explain(_analysis_or_404(analysis_id), account, _artifacts_if_ready())
    if result is None:
        raise HTTPException(status_code=404, detail=f"Account '{account}' is not in this dataset.")
    return explainer.ai_narrative(result, question)


@app.get("/api/analysis/{analysis_id}/neighbourhood")
def account_neighbourhood(analysis_id: str, account: str):
    result = explainer.neighbourhood(_analysis_or_404(analysis_id), account)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Account '{account}' is not in this dataset.")
    return result
