# FRECTION — how it works and why

A module-by-module walkthrough of the codebase: what each part does, why it is
built that way, and where its limits are.

## 1. What the app does

You upload a CSV. FRECTION works out what kind of data it is, builds a graph
from it, labels every account as **fraud**, **mule** or **normal**, draws the
network, and explains each verdict.

It handles two shapes of data:

| Shape | Example | How the graph is built |
|---|---|---|
| **Transactions** | sender, receiver, amount | accounts are nodes, transfers are edges |
| **Customer records** | one row per customer with feature columns | each customer is linked to its 5 most similar customers |

## 2. Request flow

```
Browser (React)                     FastAPI backend
───────────────                     ───────────────
upload CSV  ───── POST /api/analyze ──▶ ingest.read_upload      encoding + delimiter
                                        ingest.normalise_frame  tidy headers / values
                                        ingest.detect_columns   sender? receiver? label? id?
                                                │
                              ┌─────────────────┴─────────────────┐
                              ▼                                   ▼
                  main._analyze_transactions            entities.analyze_entities
                  graph rules (+ GNN lookup)            Isolation Forest + k-NN graph
                              └─────────────────┬─────────────────┘
                                                ▼
                                   explain.store_analysis   keep evidence in memory
graph + metrics ◀──────────────────────────────┘

click a node ──── GET /api/analysis/{id}/explain ──▶ explain.explain   evidence → reasons
search box   ──── GET /api/analysis/{id}/accounts
(optional)   ──── POST /api/analysis/{id}/ai ──────▶ explain.ai_narrative (Claude)
```

## 3. Backend modules

### `api/ingest.py` — understanding the file

**What it does.** Turns an arbitrary CSV into a clean table and decides what
each column means.

- `read_upload` tries UTF-8, then Windows-1252, then Latin-1, and picks the
  delimiter (`,` `;` tab `|`) that splits the header line into the most fields.
- `normalise_frame` trims header whitespace, renames blank and duplicate
  headers, strips whitespace from text values and drops empty rows and columns.
- `detect_columns` finds sender, receiver, amount and label columns by name.
  Short patterns such as `to` must match a whole word (`tokens()` splits
  `CustomerID` into `customer`, `id`), otherwise `to` would match inside
  "Cus**to**merID".
- If names don't help, `_pair_by_value_overlap` looks for two ID-like columns
  whose values overlap. Accounts that appear as both sender and receiver are a
  strong sign of a transaction ledger.
- If no sender/receiver pair exists, the file is treated as customer records.
- `binary_label` reads labels written as `1/0`, `Yes/No`, `fraud/legit`,
  `True/False`. For two-valued numeric labels with no `1` (for example `2/4`)
  the rarer value is treated as positive.

**Why this way.** Heuristics are transparent and fast, and the UI lets the user
override every choice, so a wrong guess is cheap.

**Limits.** Detection is rule-based, so unusual column names can still be
mis-read. The "rarer value is positive" rule is a guess. Only CSV is supported.

### `api/main.py` — the API and the transaction analysis

**Startup.** `lifespan` starts `_warm_up` in a background thread. It loads the
trained artifacts (graph, node mapping, embeddings) and clusters the embeddings
with MiniBatchKMeans. Cluster labels are cached on disk, keyed by the embeddings
file's size and modification time, so a restart doesn't redo the clustering.

**`_analyze_transactions`.** Every account string is mapped to an integer code
once with `pd.factorize`. All later logic runs on integer arrays, which is what
makes 100,000 rows take under 200 ms.

The labelling rules use labels where they exist and graph structure otherwise.
Nothing is matched on account names.

1. sent a transaction labelled fraud → fraud
2. received a transaction labelled fraud → mule
3. **collection hub** → mule. An account is a hub when
   - it receives from many distinct accounts (0.5 % of all accounts, never
     fewer than 3, and 50 is always enough),
   - it forwards to between 1 and 3 accounts, so pure sinks such as shops are
     excluded, and
   - most of its senders are *feeders*: accounts that themselves receive money
     and send most of what they send into the hub. "Most" is measured in money
     when an amount column exists, otherwise in counterparties.
4. **layering** → mule: an account most of whose senders are mules. This is
   repeated for up to 4 hops, which follows hub → shell → offshore.
5. **fraud actor**: a feeder of a confirmed hub.

The third hub condition is what separates a mule hub from a landlord or a
business. A landlord also collects from many and pays few, but the tenants spend
most of their money elsewhere, so they are not feeders.

After the rules, `api/gnn.py` scores every account with the inductive GraphSAGE
model (see section 5). The score is stored as evidence and shown in the
investigator. It does not change a verdict.

There is also a legacy path: if artifacts from the old full-PaySim pipeline are
present in `data/processed`, accounts found in that training graph take their
label from it. Those artifacts are not in the repository, so this path is
normally inactive.

**How well the rules do.** `tests/rule_benchmark.py` scores them on synthetic
ledgers with known ground truth (see `reports/rule_benchmark.md`). On the
"messy" set, which has neutral account names, shops, landlords, franchise
outlets and fraud actors who also spend at shops, precision is 100 % and recall
96 %.

**Limits — be clear about these.**

- The scores above are on synthetic data built from the same idea of a ring as
  the rules. They show the rules do what they were designed to do. They say
  nothing about real bank data.
- A ring with no fraud-actor layer, where victims pay the hub directly, is
  missed: structurally it is the same as a legitimate collector.
- Anyone who feeds a hub is called a fraud actor. A victim who sends all their
  money through an intermediary would be mislabelled.
- The thresholds (0.5 %, 3, 50, "most" = half, 4 hops) are chosen by hand, not
  learned.
- The GNN score is supporting evidence only. On a new dataset the verdicts come
  from the rules.
- Only the first 600 flagged and 200 normal transactions are drawn.

### `api/entities.py` — customer-records mode

**What it does.**

1. `_feature_matrix` builds numeric features: parses money and percent text,
   one-hot encodes text columns with at most 30 distinct values, fills gaps with
   the median, and scales each column by `(x − median) / IQR`, clipped to ±10.
2. **Isolation Forest** (100 trees) scores how unusual each customer is.
3. **k-nearest neighbours** (k = 5) links each customer to its most similar
   customers. For more than 5,000 rows, neighbours are searched in a 5,000-row
   reference sample that keeps the positive cases.
4. Labelling:
   - with a label column: fraud = labelled positive; mule = unlabelled, and
     either in the top 1 % most unusual or with at least 3 of 5 nearest
     neighbours labelled fraud
   - without a label: fraud = top 1 % most unusual; mule = the next 2 %

**Why this way.** Median and IQR scaling is robust to the extreme values that
fraud data is full of. Isolation Forest needs no labels. The neighbour rule is
"guilt by association": an unlabelled customer who looks like known fraud cases
deserves a look.

**Limits.** The 1 % and 3 % cut-offs are fixed, not learned. Euclidean distance
on mixed one-hot and numeric features is crude. With a balanced label (for
example credit approval, about 45 % positive) "fraud" simply mirrors the label.

### `api/explain.py` — the account investigator

**What it does.** After each analysis, `store_analysis` keeps the evidence
arrays in memory (the 8 most recent analyses). `explain()` then reports which
rules fired for one account, with the numbers behind them:

- transactions: which rule or cascade applied, unique senders and receivers,
  totals, top counterparties, and the GNN cluster if the account is known
- customer records: anomaly rank, how many nearest neighbours are fraud, and the
  features furthest from the median

`neighbourhood()` returns an account's direct connections so the UI can add an
account that wasn't in the initial drawing.

`ai_narrative()` is optional. If `ANTHROPIC_API_KEY` is set it sends the
evidence for that one account to Claude and returns a written summary. The
uploaded file is never sent.

**Why this way.** The verdicts come from explicit rules and scores, so the
explanation can be exact rather than approximated. That is why no model such as
SHAP is needed here.

**Limits.** Evidence lives in process memory: it is lost on restart and would
not work across several server processes.

## 4. Frontend (`dashboard/frontend/src`)

| File | Role |
|---|---|
| `App.jsx` | routes; the dashboard is lazy-loaded so the landing page stays light |
| `LandingPage.jsx` | marketing page |
| `Dashboard.jsx` | upload, progress, metric cards, column editor, layout |
| `FraudNetworkGraph.jsx` | force-directed graph on canvas (`react-force-graph-2d`) |
| `AccountInvestigator.jsx` | search, flagged-account dropdown, evidence panel, AI section |
| `AnoAI.jsx` | animated shader background (three.js) |

Points worth knowing:

- The graph keeps node objects in a cache keyed by ID, so adding an account
  later does not reshuffle the layout.
- Per-node `shadowBlur` was replaced with a translucent halo because canvas
  shadows are slow.
- The background shader renders at half resolution and pauses when the tab is
  hidden.
- Uploads use `XMLHttpRequest` rather than `fetch` because `fetch` cannot report
  upload progress.

## 5. The GNN (`src/`, `api/gnn.py`)

### `src/features.py` — leak-free features

Thirteen features per account, from structure and amounts only: transaction
counts, distinct counterparties, totals, means and maxima sent and received,
and three ratios. Amounts are log-scaled and every feature is standardised
**within the graph it was computed on**, so the model sees "large for this
dataset" rather than a currency. The same function is used in training and in
the app, so the two cannot drift apart.

No label enters a feature. An earlier pipeline used per-account fraud counts as
features while predicting fraud, which leaked the answer; that is why its
training loss fell to almost zero.

### `src/training/train_inductive.py` — training and benchmark

- **Task:** predict which accounts receive fraud-labelled money.
- **Inductive setup:** PaySim is cut into time windows of about a million
  transactions, and each window is its own graph. The model trains on two early
  windows, early-stops on a third, and is tested on later windows it has never
  seen. A freshly uploaded ledger is the same situation.
- **Model:** `NodeRiskModel` in `src/models/graphsage.py`, a 2-layer GraphSAGE
  encoder with a linear head. GraphSAGE learns how to aggregate a node's
  neighbours, not an embedding per node, which is what lets it score new graphs.
- **Baselines:** XGBoost on exactly the same features, and a one-feature
  heuristic (largest amount received).

Result: GraphSAGE reaches PR-AUC 0.021 on the test window and 0.132 on the late
window, against 0.016 and 0.094 for XGBoost. The graph helps by 35–40 %, and the
absolute numbers are low. See `reports/paysim_inductive.md`.

### `api/gnn.py` — running it on uploads

Loads `models/graphsage_inductive.pt` once at startup, computes the features for
the uploaded graph and returns a risk score per account. If PyTorch or the model
file is missing, it returns nothing and the app carries on.

**Limits.**

- What the model learned is PaySim's pattern: a fraud receiver takes one
  unusually large transfer. It ranks cash-out accounts in the top 1 % of the
  synthetic ring ledgers but does **not** rank collection hubs highly, because
  in PaySim an account with many receipts is ordinary.
- PaySim has no ring structure: every sending account appears exactly once. It
  is the wrong dataset for learning rings.
- For those two reasons the score is shown as evidence and does not decide.

### Also in `src/`

- `preprocessing/` and `training/train_gnn.py`: the original pipeline that
  builds one graph from all of PaySim and trains a transaction-level (edge)
  classifier. The two leaky features have been removed from
  `preprocessing/node_features.py`; its artifacts have not been regenerated.
- `legacy/`: earlier experiments (ANN, LSTM, DBSCAN) that nothing uses.

## 6. Evaluation

`src/evaluation/evaluate_elliptic.py` benchmarks the GraphSAGE encoder on the
Elliptic Bitcoin dataset against logistic regression, random forest, XGBoost and
an MLP, using a temporal split (train on time steps 1–29, validate on 30–34,
test on 35–49), a threshold chosen on validation, and three seeds.

Results: random forest 0.79 illicit F1, GraphSAGE 0.57. See `reports/elliptic_results.md`.

## 7. Running it

```
pip install -r requirements.txt
cd dashboard/frontend && npm install && cd ../..
.\start.ps1          # backend on :8000, frontend on :5173
```
