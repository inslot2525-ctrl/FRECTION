"""
api/explain.py
--------------
"Why is this account flagged?" — the account investigator.

1. Evidence-based explanations (always on, no API key, nothing leaves the machine)
   Every verdict in FRECTION comes from concrete rules / model signals, so we
   report exactly which ones fired for an account, with the numbers behind them.

2. Optional AI narrative (Claude)
   If ANTHROPIC_API_KEY is set (environment or a .env file in the project root),
   the evidence is sent to Claude, which writes an investigator-style summary and
   can answer follow-up questions. Claude only sees the evidence for the one
   account being explained — never the uploaded file.
"""

import os
import threading
import uuid
from collections import OrderedDict

import numpy as np

MAX_ANALYSES = 8           # most recent uploads kept in memory for investigation
TOP_COUNTERPARTIES = 8
TOP_FEATURES = 5

AI_MODEL = "claude-opus-5"

_analyses: OrderedDict = OrderedDict()
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Analysis store
# ---------------------------------------------------------------------------

def store_analysis(mode: str, mapping: dict, evidence: dict) -> str:
    analysis_id = uuid.uuid4().hex[:12]
    if mode == "transactions":
        lookup = {acct: i for i, acct in enumerate(evidence["nodes"])}
        names  = evidence["nodes"].astype(str)
    else:
        lookup = {acct: i for i, acct in enumerate(evidence["ids"])}
        names  = evidence["ids"].astype(str)
    entry = {"mode": mode, "mapping": mapping, "ev": evidence, "lookup": lookup,
             "names": np.asarray(names, dtype=object), "names_lower": np.char.lower(np.asarray(names, dtype=str))}
    with _lock:
        _analyses[analysis_id] = entry
        while len(_analyses) > MAX_ANALYSES:
            _analyses.popitem(last=False)
    return analysis_id


def get_analysis(analysis_id: str) -> dict | None:
    with _lock:
        entry = _analyses.get(analysis_id)
        if entry:
            _analyses.move_to_end(analysis_id)
        return entry


def list_accounts(entry: dict, query: str = "", limit: int = 50, group: str = "") -> dict:
    """Flagged accounts (most suspicious first), or a name search over every account.
    `group` ("fraud" / "mule" / "normal") narrows either listing."""
    ev, names = entry["ev"], entry["names"]
    groups = ev["groups"]
    if query:
        q = query.strip().lower()
        match = np.char.find(entry["names_lower"], q) >= 0
        if group:
            match &= groups == group
        hits = np.flatnonzero(match)
        # exact match, then prefix matches, then the rest
        starts = entry["names_lower"][hits]
        order = np.lexsort((starts, ~np.char.startswith(starts, q), starts != q))
        idx = hits[order][:limit]
        total = len(hits)
    else:
        flagged = np.flatnonzero(groups == group if group else groups != "normal")
        if entry["mode"] == "transactions":
            activity = ev["fan_in"][flagged] + ev["fan_out"][flagged]
        else:
            activity = ev["score"][flagged]
        order = np.lexsort((-activity, groups[flagged] != "fraud"))
        idx = flagged[order][:limit]
        total = len(flagged)
    return {"total": int(total), "accounts": [{"id": str(names[i]), "group": str(groups[i])} for i in idx]}


# ---------------------------------------------------------------------------
# Evidence → explanation
# ---------------------------------------------------------------------------

def _fmt_num(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    x = float(x) + 0.0          # + 0.0 turns -0.0 into 0.0
    if abs(x) < 0.005:
        x = 0.0
    if abs(x) >= 1000 or x == int(x):
        return f"{x:,.0f}"
    return f"{x:,.2f}"


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text[:2] != text[:2].upper() else text


def _pct(x: float) -> str:
    return f"{x:.2%}" if x > 0.999 else f"{x:.1%}"


def _names(ids, limit=3) -> str:
    ids = [str(i) for i in ids]
    shown = ", ".join(ids[:limit])
    return shown + (f" and {len(ids) - limit} more" if len(ids) > limit else "")


def _counterparties(ev, rows, other_codes, relation):
    codes = other_codes[rows]
    ok = codes >= 0
    codes, rows = codes[ok], rows[ok]
    if not len(codes):
        return []
    uniq, inv, counts = np.unique(codes, return_inverse=True, return_counts=True)
    amounts = ev["amounts"]
    totals = np.bincount(inv, weights=np.nan_to_num(amounts[rows])) if amounts is not None else None
    order = np.argsort(-counts, kind="stable")[:TOP_COUNTERPARTIES]
    return [{
        "id": str(ev["nodes"][uniq[k]]), "group": str(ev["groups"][uniq[k]]), "relation": relation,
        "transactions": int(counts[k]), "amount": None if totals is None else round(float(totals[k]), 2),
    } for k in order]


def _gnn_reason(i, ev, artifacts):
    pos = int(ev["gnn_pos"][i])
    if pos < 0 or not artifacts:
        return None
    node_idx = artifacts["node_to_idx"].get(str(ev["nodes"][i]), artifacts["node_to_idx"].get(ev["nodes"][i]))
    if node_idx is None:
        return None
    if node_idx in artifacts["known_fraud_set"]:
        return ("Known fraud in the trained GNN graph",
                "This account took part in fraudulent transactions in the historical data the Graph Neural Network was trained on.")
    cid   = int(artifacts["cluster_labels"][node_idx])
    size  = max(len(artifacts["cluster_members"][cid]), 1)
    ratio = artifacts["cluster_fraud_counts"].get(cid, 0) / size
    if ratio > 0.1:
        return ("GNN embedding sits in a fraud-heavy cluster",
                f"The GNN places this account in behaviour cluster #{cid}, where {ratio:.0%} of the {size:,} accounts are known "
                f"fraud (anything above 10% is treated as a mule ring).")
    return ("GNN embedding looks normal",
            f"The GNN places this account in behaviour cluster #{cid}, where only {ratio:.0%} of accounts are known fraud.")


def _explain_transaction(entry, i, artifacts):
    ev, mapping = entry["ev"], entry["mapping"]
    acct, group = str(ev["nodes"][i]), str(ev["groups"][i])
    sc, dc = ev["sc"], ev["dc"]
    rows_out, rows_in = np.flatnonzero(sc == i), np.flatnonzero(dc == i)
    label_col = mapping.get("fraud")

    fraud_out = int(ev["fraud_row"][rows_out].sum())
    fraud_in  = int(ev["fraud_row"][rows_in].sum())
    senders   = sc[rows_in][sc[rows_in] >= 0]
    receivers = dc[rows_out][dc[rows_out] >= 0]

    reasons = []   # (title, detail, kind)  kind: fraud | mule | info
    upper = acct.upper()
    if ev["name_frd"][i]:
        reasons.append(("Account name says FRAUD", f'The account ID "{acct}" contains the word FRAUD.', "fraud"))
    if ev["name_mule"][i]:
        word = next(w for w in ("MULE", "OFFSHORE", "SHELL") if w in upper)
        reasons.append((f"Account name suggests a {word.lower()} account",
                        f'The account ID contains "{word}", a pattern used for mule, offshore and shell-company accounts.', "mule"))
    if ev["is_fs"][i]:
        reasons.append(("Sent fraud-labelled transactions",
                        f"{fraud_out} of its {len(rows_out)} outgoing transactions are marked as fraud in the '{label_col}' column.", "fraud"))
    if ev["is_fr"][i]:
        reasons.append(("Received fraud-labelled transactions",
                        f"{fraud_in} of its {len(rows_in)} incoming transactions are marked as fraud in the '{label_col}' column — "
                        "the account is on the receiving end of fraud, typical of a mule.", "mule"))
    if ev["fan_rule"][i]:
        reasons.append(("Collection-hub pattern (fan-in)",
                        f"Received money from {int(ev['fan_in'][i])} different senders (the threshold for this file is "
                        f"{ev['fan_thresh']:.0f}) but sends to only {int(ev['fan_out'][i])} account(s). Many-in / few-out is how "
                        "mule accounts consolidate stolen funds.", "mule"))
    if ev["struct_mule"][i] and not ev["base_mule"][i]:
        upstream = np.unique(senders[ev["base_mule"][senders]])
        reasons.append(("Receives money from a mule",
                        f"Gets funds from mule account(s) {_names(ev['nodes'][upstream])}, so it is the next hop in the "
                        "laundering chain (layering / cash-out).", "mule"))
    if ev["struct_fraud"][i] and not ev["base_fraud"][i]:
        downstream = np.unique(receivers[ev["struct_mule"][receivers]])
        reasons.append(("Feeds money into a mule",
                        f"Sends funds to mule account(s) {_names(ev['nodes'][downstream])}. Accounts that pay into a mule hub "
                        "are treated as the fraud actors supplying the ring.", "fraud"))

    gnn = _gnn_reason(i, ev, artifacts)
    structural = "fraud" if ev["struct_fraud"][i] else "mule" if ev["struct_mule"][i] else "normal"
    if gnn:
        reasons.append((gnn[0], gnn[1], group if group != "normal" else "info"))
        if structural != group:
            reasons.append(("GNN verdict takes priority",
                            f"Graph rules alone would say '{structural}', but this account is known to the trained GNN, "
                            f"so its verdict ('{group}') is used.", "info"))

    linked_flagged = np.unique(np.concatenate([senders, receivers]))
    linked_flagged = linked_flagged[ev["groups"][linked_flagged] != "normal"]

    if group == "normal" and structural != "normal":
        summary = (f"{acct} is not flagged. The graph rules alone would mark it as '{structural}', but the trained GNN "
                   "knows this account and classifies its behaviour as normal, so the GNN verdict is used.")
    elif group == "normal":
        summary = (f"{acct} is not flagged. None of the fraud signals fired: no fraud-labelled transactions, no suspicious "
                   f"name, no collection-hub pattern and no money flowing to or from a mule.")
        if len(linked_flagged):
            summary += f" It does transact with {len(linked_flagged)} flagged account(s), which may be worth a look."
    else:
        role = "a fraud actor" if group == "fraud" else "a money mule"
        main = [r for r in reasons if r[2] == group] or reasons
        summary = f"{acct} is flagged as {role} because: " + "; ".join(_lower_first(r[0]) for r in main) + "."

    stats = [
        {"label": "Outgoing transactions", "value": f"{len(rows_out):,} to {int(ev['fan_out'][i]):,} accounts"},
        {"label": "Incoming transactions", "value": f"{len(rows_in):,} from {int(ev['fan_in'][i]):,} accounts"},
    ]
    if ev["amounts"] is not None:
        stats += [
            {"label": "Total sent", "value": _fmt_num(np.nansum(ev["amounts"][rows_out]))},
            {"label": "Total received", "value": _fmt_num(np.nansum(ev["amounts"][rows_in]))},
        ]
    if ev["has_label"]:
        stats.append({"label": "Fraud-labelled transactions", "value": f"{fraud_out} sent · {fraud_in} received"})
    stats.append({"label": "Linked flagged accounts", "value": f"{len(linked_flagged):,}"})

    counterparties = _counterparties(ev, rows_out, dc, "sent to") + _counterparties(ev, rows_in, sc, "received from")
    counterparties.sort(key=lambda c: (c["group"] == "normal", -c["transactions"]))

    return {
        "account": acct, "group": group, "mode": "transactions", "summary": summary,
        "reasons": [{"title": t, "detail": d, "kind": k} for t, d, k in reasons],
        "stats": stats, "counterparties": counterparties[:TOP_COUNTERPARTIES * 2],
    }


def _explain_entity(entry, i):
    ev, mapping = entry["ev"], entry["mapping"]
    acct, group = str(ev["ids"][i]), str(ev["groups"][i])
    n = len(ev["ids"])
    rank = int(ev["rank"][i])
    more_unusual_than = 1 - (rank + 1) / n
    reasons = []

    if ev["labelled"] and ev["label"][i]:
        reasons.append(("Labelled fraud in the data",
                        f"The '{mapping.get('fraud')}' column marks this customer as positive/fraud.", "fraud"))

    if ev["top_anom"][i]:
        reasons.append(("Among the most unusual customers",
                        f"Isolation Forest ranks it #{rank + 1:,} of {n:,} — more unusual than {_pct(more_unusual_than)} of "
                        f"customers (top {ev['contamination']:.0%} are flagged).", "fraud" if not ev["labelled"] else "mule"))
    elif ev["band_anom"][i] and not ev["labelled"]:
        reasons.append(("Borderline unusual",
                        f"Ranked #{rank + 1:,} of {n:,} by Isolation Forest — inside the top {ev['suspect_band']:.0%} "
                        f"band, just below the most extreme {ev['contamination']:.0%}.", "mule"))

    nbrs = ev["nbrs"][i]
    nbrs = nbrs[nbrs >= 0]
    peers = [{"id": str(ev["ids"][j]), "group": str(ev["groups"][j]), "relation": "look-alike",
              "labelled_fraud": bool(ev["label"][j]) if ev["labelled"] else None} for j in nbrs]
    if ev["peer_frac"] is not None and len(nbrs):
        k_fraud = int(round(ev["peer_frac"][i] * len(nbrs)))
        if ev["peer_frac"][i] >= ev["peer_threshold"] and not ev["label"][i]:
            reasons.append(("Looks like known fraud cases",
                            f"{k_fraud} of its {len(nbrs)} most similar customers are labelled fraud, although this one "
                            "isn't — a classic sign of an undiscovered member of the same ring.", "mule"))

    # Which features make it stand out (robust z-score against the typical customer)
    unusual = []
    if len(ev["feature_names"]):
        z = (ev["X_raw"][i] - ev["medians"]) / ev["scales"]
        for f in np.argsort(-np.abs(z))[:TOP_FEATURES]:
            if abs(z[f]) < 1.5:
                break
            unusual.append({
                "feature": ev["feature_names"][f], "value": _fmt_num(ev["X_raw"][i][f]),
                "typical": _fmt_num(ev["medians"][f]), "direction": "higher" if z[f] > 0 else "lower",
                "strength": round(float(abs(z[f])), 1),
            })
    if unusual and group != "normal":
        parts = [f"{u['feature']} = {u['value']} (typical {u['typical']})" for u in unusual[:3]]
        reasons.append(("What stands out", "Values far from the typical customer: " + "; ".join(parts) + ".", "info"))

    if group == "normal":
        summary = (f"{acct} is not flagged: it ranks #{rank + 1:,} of {n:,} for unusualness"
                   + (" and is not labelled fraud" if ev["labelled"] else "") + ".")
    else:
        role = "fraud" if group == "fraud" else "a suspected look-alike"
        main = [r for r in reasons if r[2] in (group, "fraud", "mule")] or reasons
        summary = f"{acct} is flagged as {role} because: " + "; ".join(_lower_first(r[0]) for r in main) + "."

    stats = [{"label": "Unusualness rank", "value": f"#{rank + 1:,} of {n:,}"},
             {"label": "More unusual than", "value": f"{_pct(more_unusual_than)} of customers"}]
    if ev["labelled"]:
        stats.append({"label": f"Label ({mapping.get('fraud')})", "value": "fraud / positive" if ev["label"][i] else "not fraud"})
    if ev["row_counts"] is not None and ev["row_counts"][i] > 1:
        stats.append({"label": "Rows merged", "value": f"{int(ev['row_counts'][i])} rows for this customer"})

    return {
        "account": acct, "group": group, "mode": "entities", "summary": summary,
        "reasons": [{"title": t, "detail": d, "kind": k} for t, d, k in reasons],
        "stats": stats, "unusual_features": unusual, "counterparties": peers,
    }


def explain(entry: dict, account: str, artifacts: dict | None) -> dict | None:
    i = entry["lookup"].get(account)
    if i is None:
        return None
    if entry["mode"] == "transactions":
        return _explain_transaction(entry, i, artifacts)
    return _explain_entity(entry, i)


# ---------------------------------------------------------------------------
# Optional AI narrative (Claude)
# ---------------------------------------------------------------------------

def _load_dotenv():
    """Tiny .env loader (KEY=value lines) so users can drop their key in a file."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def ai_available() -> bool:
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


SYSTEM_PROMPT = """You are the explanation assistant inside FRECTION, a fraud-ring detection tool used by \
analysts. You receive the evidence FRECTION computed for ONE account and explain it to a non-technical analyst.

Rules:
- Use only the evidence provided. Never invent transactions, amounts, people or facts.
- These are risk signals from rules and models, not proof of wrongdoing — say so briefly.
- Plain English, no jargon without a short explanation. Keep it under about 180 words.
- Format: one-sentence verdict, then 2-4 short bullets on why, then one bullet "What to check next" with a concrete step.
- If the account is not flagged, explain why it looks normal and whether any link deserves a look.
- When answering a follow-up question, answer it directly from the evidence; if the evidence can't answer it, say so."""


def ai_narrative(explanation: dict, question: str | None = None) -> dict:
    import json

    import anthropic

    evidence = json.dumps(explanation, ensure_ascii=False, indent=1)
    ask = question.strip() if question else "Explain why this account received its verdict."
    client = anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model=AI_MODEL,
            max_tokens=4000,
            output_config={"effort": "low"},        # short, evidence-bound write-up: low effort keeps it fast
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",                     # a declined request is retried on the recommended model
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"<evidence>\n{evidence}\n</evidence>\n\n{ask}"}],
        )
    except anthropic.AuthenticationError:
        return {"ok": False, "error": "The Anthropic API key was rejected. Check ANTHROPIC_API_KEY in your .env file."}
    except anthropic.PermissionDeniedError:
        return {"ok": False, "error": "This API key isn't allowed to use the model."}
    except anthropic.RateLimitError:
        return {"ok": False, "error": "Rate limited by the Anthropic API — try again in a minute."}
    except anthropic.APIStatusError as e:
        return {"ok": False, "error": f"Anthropic API error ({e.status_code}). Try again shortly."}
    except anthropic.APIConnectionError:
        return {"ok": False, "error": "Couldn't reach the Anthropic API — check your internet connection."}

    if response.stop_reason == "refusal":
        return {"ok": False, "error": "The AI declined to write this explanation. The evidence above still applies."}
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return {"ok": True, "text": text, "model": response.model}


# ---------------------------------------------------------------------------
# Neighbourhood: an account's direct connections, so the UI can add it to the
# graph when it wasn't part of the initial drawing.
# ---------------------------------------------------------------------------

MAX_NEIGHBOURHOOD_LINKS = 60


def neighbourhood(entry: dict, account: str) -> dict | None:
    i = entry["lookup"].get(account)
    if i is None:
        return None
    ev, names, groups = entry["ev"], entry["names"], entry["ev"]["groups"]
    if entry["mode"] == "transactions":
        sc, dc = ev["sc"], ev["dc"]
        rows = np.flatnonzero(((sc == i) | (dc == i)) & (sc >= 0) & (dc >= 0))
        other = np.where(sc[rows] == i, dc[rows], sc[rows])
        # flagged counterparties first so the interesting links survive the cap
        rows = rows[np.argsort(groups[other] == "normal", kind="stable")][:MAX_NEIGHBOURHOOD_LINKS]
        pairs = list(dict.fromkeys(zip(sc[rows].tolist(), dc[rows].tolist())))
    else:
        nbrs = ev["nbrs"][i]
        pairs = [(i, int(j)) for j in nbrs[nbrs >= 0]]
    idx = sorted({i} | {a for p in pairs for a in p})
    return {
        "nodes": [{"id": str(names[k]), "group": str(groups[k])} for k in idx],
        "links": [{"source": str(names[a]), "target": str(names[b])} for a, b in pairs],
    }
