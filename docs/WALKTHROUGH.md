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

The labelling rules, in priority order:

1. account name contains `FRAUD` → fraud
2. account name contains `MULE`, `OFFSHORE` or `SHELL` → mule
3. sent a transaction labelled fraud → fraud
4. received a transaction labelled fraud → mule
5. **fan-in rule**: received from many distinct senders (at least 0.5 % of all
   accounts, minimum 2) while sending to at most 3 → mule
6. cascade: anything a mule pays is also a mule
7. cascade: anything that pays a mule is a fraud actor

If the trained GNN artifacts are present, accounts that appear in the training
graph take their label from there instead: known fraud stays fraud, and an
account in an embedding cluster where more than 10 % of members are known fraud
is a mule.

**Why this way.** The rules encode the textbook shape of a laundering ring:
many victims pay fraud actors, who pay a collection hub, which pays a shell
company, which pays offshore.

**Limits — be clear about these.**

- The GNN is a **lookup of accounts seen in training**. It does not run on
  newly uploaded accounts. On a new dataset the verdicts come from the rules.
- Rules 1 and 2 match account *names*. They work on the synthetic demo data and
  would do nothing on real data.
- The fan-in rule flags any popular receiver, including ordinary shops, and the
  cascade then flags everyone who paid them. On the bundled sample this reports
  878 fraud accounts where 150 are planted.
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

## 5. Training pipeline (`src/`)

| Stage | File | Output |
|---|---|---|
| map accounts to integer IDs | `preprocessing/node_mapping.py` | `edge_index.pt`, `edge_attr.pt` |
| node features | `preprocessing/node_features.py` | `x_node_features.pt` |
| assemble graph | `preprocessing/build_pyg_graph.py` | `pyg_graph.pt` |
| model | `models/graphsage.py` | 2-layer GraphSAGE encoder + MLP edge decoder |
| train | `training/train_gnn.py` | `models_gnn.pth` |
| embeddings | `inference/extract_gnn_embeddings.py` | `gnn_embeddings.pt` |

The model predicts whether a **transaction (edge)** is fraud from the embeddings
of its two accounts. Training uses class-weighted binary cross-entropy and early
stopping on validation PR-AUC.

**Known problem — label leakage.** `node_features.py` includes `fraud_sent` and
`fraud_received`: the number of fraud-labelled transactions per account. The
model is then asked to predict whether a transaction is fraud. Most PaySim
accounts appear once, so these features contain the answer. Scores from this
pipeline cannot be trusted until those two features are removed.

## 6. Evaluation

`src/evaluation/evaluate_elliptic.py` benchmarks the GraphSAGE encoder on the
Elliptic Bitcoin dataset against logistic regression, random forest, XGBoost and
an MLP, using a temporal split (train on time steps 1–29, validate on 30–34,
test on 35–49), a threshold chosen on validation, and three seeds.

Results: see `reports/elliptic_results.md` (generated by the script).

## 7. Running it

```
pip install -r requirements.txt
cd dashboard/frontend && npm install && cd ../..
.\start.ps1          # backend on :8000, frontend on :5173
```
