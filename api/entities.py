"""
api/entities.py
---------------
Fraud analysis for datasets with ONE ROW PER CUSTOMER / ACCOUNT (no sender →
receiver transactions), e.g. credit-approval or card-fraud feature tables.

Pipeline
--------
1. Normalise features: parse numeric text ("$1,200", "12%"), one-hot small
   categoricals, fill gaps with medians, robust-scale every column.
2. Isolation Forest (100 trees, 1 % contamination) scores how unusual each
   customer is.
3. A k-nearest-neighbour similarity graph links every customer to the
   customers that look most like them — fraud rings show up as tight clusters.
4. Grouping
   * with a label column:   fraud  = labelled positive
                            mule   = unlabelled but anomalous, or whose closest
                                     look-alikes are mostly fraud
   * without a label:       fraud  = top 1 % most anomalous
                            mule   = next 2 % (borderline anomalies)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

from api.ingest import binary_label, is_id_like_name, is_text, parse_numeric_text

CONTAMINATION   = 0.01
SUSPECT_BAND    = 0.03
K_NEIGHBOURS    = 5
PEER_THRESHOLD  = 0.6      # ≥ 3 of 5 nearest look-alikes are fraud
MAX_REFERENCE   = 5_000    # rows used as the neighbour search index
MAX_CATEGORIES  = 30
VIS_FOCUS_NODES = 350
VIS_NORMAL_SEED = 120
VIS_EDGES_PER   = 3


def _feature_matrix(df: pd.DataFrame, skip: set):
    """Returns (scaled X, feature names, raw imputed X, medians, scales)."""
    parts, names = [], []
    n = len(df)
    for c in df.columns:
        if c in skip:
            continue
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            s = s.astype(float)
        if is_text(s):
            parsed = parse_numeric_text(s)
            if parsed.notna().sum() >= 0.9 * max(s.notna().sum(), 1):
                s = parsed
            elif s.nunique(dropna=True) <= MAX_CATEGORIES:
                dummies = pd.get_dummies(s, prefix=c, dtype=float)
                parts.append(dummies.to_numpy())
                names.extend(dummies.columns.astype(str))
                continue
            else:
                continue                       # free text / secondary IDs
        if not pd.api.types.is_numeric_dtype(s):
            continue
        if is_id_like_name(c) and s.nunique(dropna=True) > 0.9 * n:
            continue                           # e.g. a second account-number column
        if s.nunique(dropna=True) <= 1:
            continue                           # constant — carries no signal
        parts.append(s.to_numpy(dtype=float).reshape(-1, 1))
        names.append(c)

    if not parts:
        return np.empty((n, 0)), [], np.empty((n, 0)), np.empty(0), np.empty(0)

    X = np.hstack(parts)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    X = np.where(np.isnan(X), med, X)

    q75, q25 = np.percentile(X, [75, 25], axis=0)
    scale = q75 - q25
    std = X.std(axis=0)
    scale = np.where(scale > 0, scale, np.where(std > 0, std, 1.0))
    X_scaled = np.clip((X - med) / scale, -10, 10)
    return X_scaled.astype(np.float32), names, X, med, scale


def analyze_entities(df: pd.DataFrame, id_col, label_col, amount_col, as_str_ids):
    skip = {c for c in (id_col, label_col) if c}

    if id_col:
        ids = as_str_ids(df[id_col])
        ids = ids.fillna(pd.Series([f"Row {i + 2}" for i in range(len(df))], index=df.index))
    else:
        ids = pd.Series([f"Row {i + 2}" for i in range(len(df))], index=df.index)   # +2 = spreadsheet row

    X, feature_names, X_raw, medians, scales = _feature_matrix(df, skip)
    if X.shape[1] == 0:
        raise ValueError(
            "Couldn't find any numeric or categorical columns to analyse. "
            "Upload either transactions (sender → receiver) or one row per customer with feature columns."
        )

    y = binary_label(df[label_col]) if label_col else None

    # One node per customer: merge repeated rows for the same ID
    if ids.duplicated().any():
        keys   = ids.to_numpy()
        merged = pd.DataFrame(X).groupby(keys, sort=False).mean()
        X      = merged.to_numpy(dtype=np.float32)
        X_raw  = pd.DataFrame(X_raw).groupby(keys, sort=False).mean().to_numpy()
        row_counts = pd.Series(keys).groupby(keys, sort=False).size().to_numpy()
        if y is not None:
            y = pd.Series(y).groupby(ids.to_numpy(), sort=False).max().to_numpy()
        ids = pd.Series(merged.index.astype(str))
    else:
        row_counts = None
    ids = ids.astype(str).to_numpy()
    n = len(ids)

    # 1. Anomaly scores
    forest = IsolationForest(n_estimators=100, contamination=CONTAMINATION, random_state=42, n_jobs=-1)
    forest.fit(X)
    score = -forest.score_samples(X)                       # higher = more unusual
    order = np.argsort(-score, kind="stable")
    rank  = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    top_anom  = rank < max(1, int(np.ceil(CONTAMINATION * n)))
    band_anom = rank < max(1, int(np.ceil(SUSPECT_BAND * n)))

    # 2. Similarity graph (neighbour index on a representative sample for speed)
    rng = np.random.default_rng(42)
    if n <= MAX_REFERENCE:
        ref = np.arange(n)
    else:
        pos = np.flatnonzero(y) if y is not None else np.flatnonzero(top_anom)
        pos = rng.choice(pos, min(len(pos), MAX_REFERENCE // 2), replace=False) if len(pos) else pos
        rest = np.setdiff1d(np.arange(n), pos)
        ref = np.sort(np.concatenate([pos, rng.choice(rest, MAX_REFERENCE - len(pos), replace=False)]))

    k = min(K_NEIGHBOURS + 1, len(ref))
    nn = NearestNeighbors(n_neighbors=k).fit(X[ref])
    _, idx = nn.kneighbors(X)
    nbrs = ref[idx]                                        # global row indices
    # Drop each row's match with itself (move it to the end), keep the K closest others
    self_hit = nbrs == np.arange(n)[:, None]
    order    = np.argsort(self_hit, axis=1, kind="stable")
    nbrs     = np.take_along_axis(nbrs, order, axis=1)
    nbrs     = np.where(np.take_along_axis(self_hit, order, axis=1), -1, nbrs)[:, :K_NEIGHBOURS]

    # 3. Grouping
    if y is not None and y.any():
        fraud = y.astype(bool)
        valid = nbrs >= 0
        peer_frac = (np.where(valid, fraud[np.maximum(nbrs, 0)], False).sum(1) / np.maximum(valid.sum(1), 1))
        mule = ~fraud & (top_anom | (peer_frac >= PEER_THRESHOLD))
        engine = "isolation-forest+knn (labelled)"
    else:
        peer_frac = None
        fraud = top_anom
        mule  = band_anom & ~top_anom
        engine = "isolation-forest+knn"

    groups = np.where(fraud, "fraud", np.where(mule, "mule", "normal"))

    # 4. Visual slice: flagged customers + their closest look-alikes
    mule_idx  = np.flatnonzero(mule)
    fraud_idx = np.flatnonzero(fraud)
    mule_idx  = mule_idx[np.argsort(-score[mule_idx], kind="stable")][: VIS_FOCUS_NODES // 2]
    fraud_idx = fraud_idx[np.argsort(-score[fraud_idx], kind="stable")][: VIS_FOCUS_NODES - len(mule_idx)]
    normal_all = np.flatnonzero(groups == "normal")
    normal_idx = rng.choice(normal_all, min(len(normal_all), VIS_NORMAL_SEED), replace=False) if len(normal_all) else normal_all
    focus = np.unique(np.concatenate([mule_idx, fraud_idx, normal_idx]).astype(np.int64))

    edges = set()
    for i in focus:
        for j in nbrs[i, :VIS_EDGES_PER]:
            if j >= 0 and j != i:
                edges.add((min(i, j), max(i, j)))
    vis = np.unique(np.concatenate([focus, np.array([e for pair in edges for e in pair], dtype=np.int64)]))

    graph_data = {
        "nodes": [{"id": ids[i], "group": groups[i]} for i in vis.tolist()],
        "links": [{"source": ids[a], "target": ids[b]} for a, b in sorted(edges)],
    }
    metrics = {
        "total_nodes":      int(n),
        "total_edges":      int(len(df)),
        "known_fraudsters": int(fraud.sum()),
        "suspected_mules":  int(mule.sum()),
    }
    evidence = {
        "ids": ids, "groups": groups, "labelled": y is not None and bool(y.any()),
        "label": None if y is None else y.astype(bool), "score": score, "rank": rank,
        "top_anom": top_anom, "band_anom": band_anom, "peer_frac": peer_frac, "nbrs": nbrs,
        "X": X, "X_raw": X_raw, "medians": medians, "scales": scales,
        "feature_names": feature_names, "row_counts": row_counts,
        "contamination": CONTAMINATION, "suspect_band": SUSPECT_BAND, "peer_threshold": PEER_THRESHOLD,
    }
    return metrics, graph_data, engine, feature_names, evidence
