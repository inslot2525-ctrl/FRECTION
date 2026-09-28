"""
api/ingest.py
-------------
Turns an arbitrary uploaded CSV into a clean DataFrame and works out what the
columns mean, so users don't have to reformat their data first.

* read_upload()     encoding + delimiter detection, row cap
* normalise_frame() tidy headers, drop empty rows/columns, trim whitespace
* detect_columns()  sender / receiver / amount / label / id inference
* binary_label()    turn any label column (1/0, Yes/No, fraud/legit, ...) into bool
"""

import io
import re

import numpy as np
import pandas as pd

DELIMITERS = [",", ";", "\t", "|"]

SENDER_PATTERNS   = ["nameorig", "sender", "source", "from", "payer", "originator", "src", "acct_from", "account_from", "origin"]
RECEIVER_PATTERNS = ["namedest", "receiver", "dest", "target", "to", "payee", "beneficiary", "dst", "acct_to", "account_to", "destination"]
AMOUNT_PATTERNS   = ["amount", "amt", "value", "sum", "transaction_amount", "trans_amount", "money", "price", "total"]
LABEL_PATTERNS    = ["isfraud", "is_fraud", "fraud", "fraudulent", "label", "class", "flag", "suspicious", "target"]

# Words that mark a column as an identifier rather than a measurement
ID_TOKENS = {"id", "customer", "cust", "account", "acct", "user", "client", "card", "member",
             "key", "no", "num", "number", "code", "name", "iban", "wallet", "party", "entity"}

POSITIVE_WORDS = {"1", "true", "t", "yes", "y", "fraud", "fraudulent", "suspicious", "positive",
                  "bad", "anomaly", "anomalous", "malicious", "scam", "default", "risky", "high"}


# ---------------------------------------------------------------------------
# Reading & normalising
# ---------------------------------------------------------------------------

def _sniff_delimiter(sample: str) -> str:
    """Pick the delimiter that splits the header line into the most fields."""
    lines = [ln for ln in sample.splitlines() if ln.strip()][:5]
    if not lines:
        return ","
    counts = {d: lines[0].count(d) for d in DELIMITERS}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _detect_encoding(contents: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            contents.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            pass
    return "latin-1"                      # never fails


def read_upload(contents: bytes, max_rows: int) -> pd.DataFrame:
    """Parse the upload, tolerating BOMs, non-UTF-8 encodings and ; / tab / | delimiters."""
    encoding = _detect_encoding(contents)
    sep = _sniff_delimiter(contents[:65536].decode(encoding, errors="ignore"))
    return pd.read_csv(io.BytesIO(contents), sep=sep, nrows=max_rows + 1,
                       encoding=encoding, low_memory=False, skipinitialspace=True)


def is_text(s: pd.Series) -> bool:
    """Text column? (object in pandas 2, StringDtype in pandas 3)"""
    return s.dtype == object or pd.api.types.is_string_dtype(s)


_NUMERIC_JUNK = r"[,\s$€£₹¥%]"


def parse_numeric_text(s: pd.Series) -> pd.Series:
    """'$1,200' → 1200.0, '12%' → 12.0; anything unparseable → NaN."""
    parsed = pd.to_numeric(s.astype(str).str.replace(_NUMERIC_JUNK, "", regex=True), errors="coerce")
    return parsed.where(s.notna())


def normalise_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Tidy headers and values without changing what the data means."""
    names, seen = [], {}
    for i, c in enumerate(df.columns):
        name = str(c).strip().strip('"').strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"column_{i + 1}"
        n = seen.get(name, 0)
        seen[name] = n + 1
        names.append(name if n == 0 else f"{name}_{n + 1}")
    df.columns = names

    for c in df.columns:
        s = df[c]
        if is_text(s) and s.map(type).eq(str).sum() == s.notna().sum():   # plain text column
            df[c] = s.str.strip().replace("", np.nan)

    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Column understanding
# ---------------------------------------------------------------------------

def tokens(name: str) -> set[str]:
    """'CustomerID' -> {'customer', 'id'};  'acct_from' -> {'acct', 'from'}."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
    return {t.lower() for t in re.split(r"[^A-Za-z0-9]+", spaced) if t}


def _matches(pattern: str, col: str) -> bool:
    # Very short patterns ("to", "src", "amt") must be whole words — otherwise
    # "to" would match "CustomerID" and "sum" would match "consumer".
    if len(pattern) <= 3:
        return pattern in tokens(col)
    return pattern in str(col).lower().strip()


def find_col(columns, patterns, exclude=()):
    for pat in patterns:
        for c in columns:
            if c not in exclude and _matches(pat, c):
                return c
    return None


def is_id_like_name(col: str) -> bool:
    return bool(tokens(col) & ID_TOKENS)


def _id_values(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if pd.api.types.is_float_dtype(s) and (s == np.floor(s)).all():
        s = s.astype("int64")
    return s.astype(str)


def _looks_like_account_column(df: pd.DataFrame, c: str) -> bool:
    s = df[c]
    if pd.api.types.is_bool_dtype(s):
        return False
    nunique = s.nunique(dropna=True)
    if nunique < max(10, 0.02 * len(df)):
        return False                       # categories, not accounts
    if is_text(s):
        if s.dropna().astype(str).str.len().median() > 40:
            return False                   # free text
        if parse_numeric_text(s).notna().sum() >= 0.9 * s.notna().sum():
            return False                   # money / numbers stored as text ("€48,661")
        return True
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        return is_id_like_name(c)          # numbers only count if the name says ID
    return False


def _pair_by_value_overlap(df: pd.DataFrame, candidates: list[str]):
    """Two columns that share many values are almost certainly sender/receiver."""
    best, best_score = None, 0.05
    sample = df.head(20_000)
    values = {c: set(_id_values(sample[c])) for c in candidates}
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            inter = len(values[a] & values[b])
            if not inter:
                continue
            score = inter / len(values[a] | values[b])
            if score > best_score:
                best, best_score = (a, b), score
    return best


def detect_columns(df: pd.DataFrame) -> dict:
    """Infer the role of each column. Returns a mapping incl. the analysis mode."""
    cols = list(df.columns)
    used: set = set()

    sender = find_col(cols, SENDER_PATTERNS)
    if sender: used.add(sender)
    receiver = find_col(cols, RECEIVER_PATTERNS, exclude=used)
    if receiver: used.add(receiver)
    amount = find_col(cols, AMOUNT_PATTERNS, exclude=used)
    if amount: used.add(amount)
    label = find_col(cols, LABEL_PATTERNS, exclude=used)
    if label and df[label].nunique(dropna=True) > 10:
        label = None                       # e.g. "target_account" is not a class label

    if not sender or not receiver:
        accounts = [c for c in cols if c not in used and c != label and _looks_like_account_column(df, c)]
        pair = _pair_by_value_overlap(df, accounts)
        if pair:
            a, b = pair
            if not sender and not receiver:
                sender, receiver = a, b
            elif not receiver:
                receiver = b if a == sender else a
            else:
                sender = a if b == receiver else b
        else:
            # Two text ID columns (e.g. customer -> merchant) still form a graph
            text_ids = sorted((c for c in accounts if is_text(df[c])),
                              key=lambda c: df[c].nunique(), reverse=True)
            if not sender and text_ids:
                sender = text_ids.pop(0)
            if not receiver and text_ids:
                receiver = text_ids[0]

    if sender and receiver and sender != receiver:
        return {"mode": "transactions", "sender": sender, "receiver": receiver,
                "amount": amount, "fraud": label, "id": None}

    # No transaction pair → one row per customer / account
    id_col = detect_id_column(df, exclude={label})
    return {"mode": "entities", "sender": None, "receiver": None,
            "amount": amount if amount != id_col else None, "fraud": label, "id": id_col}


def detect_id_column(df: pd.DataFrame, exclude=frozenset()):
    best, best_ratio = None, 0.5
    for c in df.columns:
        if c in exclude or not is_id_like_name(c) or pd.api.types.is_bool_dtype(df[c]):
            continue
        ratio = df[c].nunique(dropna=True) / max(len(df), 1)
        if ratio > best_ratio:
            best, best_ratio = c, ratio
    return best


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def binary_label(col: pd.Series) -> np.ndarray:
    """Interpret a fraud / class column as a boolean 'is positive' array."""
    num = pd.to_numeric(col, errors="coerce")
    if num.notna().sum() >= 0.9 * col.notna().sum() and num.notna().any():
        positive = num.eq(1)
        values = set(num.dropna().unique())
        if not positive.any() and len(values) == 2:
            # Binary codes without a 1 (e.g. 2/4, -1/0): the rarer value is the positive class
            minority = num.value_counts().idxmin()
            positive = num.eq(minority)
        return positive.to_numpy()
    text = col.astype(str).str.strip().str.lower()
    return text.isin(POSITIVE_WORDS).to_numpy() & col.notna().to_numpy()
