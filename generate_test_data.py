"""
generate_test_data.py
---------------------
Generates a small, deterministic test CSV with KNOWN expected counts.

Expected results after upload:
  - Total Accounts Scanned : ~1,315
  - Transactions Processed : 2,800+ (approx)
  - Known Malicious Hubs   : 150    (3 rings × 50 fraudsters)
  - Newly Identified Mules : 9      (3 mule hubs + 3 offshore + 3 shell accounts)

Run:
    python generate_test_data.py
Output file:  test_ledger_small.csv
"""

import random
import sys
import pandas as pd

# Windows consoles default to cp1252, which can't print the emoji below
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

random.seed(42)  # Fixed seed → reproducible counts every run

# ── CONFIG ──────────────────────────────────────────────────────────────────
NUM_NORMAL_USERS  = 1_000
NUM_MERCHANTS     = 200
NUM_NORMAL_TX     = 2_000   # background noise
NUM_FRAUD_RINGS   = 3
FRAUDSTERS_PER_RING = 50    # exactly 50 per ring → 150 total fraudsters
VICTIMS_PER_FRAUD = 5       # victims per fraudster

# ── LEGITIMATE ACCOUNTS ─────────────────────────────────────────────────────
normal_users = [f"U_{i}"  for i in range(NUM_NORMAL_USERS)]
merchants    = [f"M_{i}"  for i in range(NUM_MERCHANTS)]

transactions = []

# ── 1. NORMAL BACKGROUND TRANSACTIONS ───────────────────────────────────────
print(f"💸 Generating {NUM_NORMAL_TX} normal transactions...")
for _ in range(NUM_NORMAL_TX):
    sender   = random.choice(normal_users)
    receiver = random.choice(merchants) if random.random() > 0.2 else random.choice(normal_users)
    amount   = round(random.uniform(5.0, 800.0), 2)
    transactions.append({"Sender_ID": sender, "Receiver_Acct": receiver, "Tx_Amount": amount})

# ── 2. FRAUD RINGS ──────────────────────────────────────────────────────────
print(f"🕷️  Injecting {NUM_FRAUD_RINGS} fraud rings ({FRAUDSTERS_PER_RING} fraudsters each)...")

total_fraudsters = 0
total_mule_nodes  = 0

for ring_id in range(NUM_FRAUD_RINGS):
    mule_hub      = f"MULE_HUB_CRITICAL_{ring_id}"      # mule
    offshore_acct = f"OFFSHORE_CAYMAN_{ring_id}"         # mule (offshore)
    shell_co      = f"SHELL_CORP_{ring_id}"              # mule (layering)
    fraudsters    = [f"FRAUD_ACT_{ring_id}_{i}" for i in range(FRAUDSTERS_PER_RING)]

    total_fraudsters += len(fraudsters)
    total_mule_nodes += 3  # mule_hub + offshore_acct + shell_co

    # Step 1 — victims → fraudsters  (harvesting)
    for f_node in fraudsters:
        for _ in range(VICTIMS_PER_FRAUD):
            victim = random.choice(normal_users)
            amount = round(random.uniform(200.0, 2_000.0), 2)
            transactions.append({"Sender_ID": victim, "Receiver_Acct": f_node, "Tx_Amount": amount})

        # Step 2 — fraudster → mule hub  (consolidation)
        transactions.append({
            "Sender_ID":    f_node,
            "Receiver_Acct": mule_hub,
            "Tx_Amount":    round(random.uniform(3_000.0, 15_000.0), 2),
        })

    # Step 3 — mule hub → shell company  (layering)
    transactions.append({
        "Sender_ID":    mule_hub,
        "Receiver_Acct": shell_co,
        "Tx_Amount":    round(random.uniform(50_000.0, 200_000.0), 2),
    })

    # Step 4 — shell company → offshore  (cash-out)
    transactions.append({
        "Sender_ID":    shell_co,
        "Receiver_Acct": offshore_acct,
        "Tx_Amount":    round(random.uniform(80_000.0, 400_000.0), 2),
    })

# ── 3. SHUFFLE & SAVE ───────────────────────────────────────────────────────
df = pd.DataFrame(transactions).sample(frac=1, random_state=42).reset_index(drop=True)

filename = "test_ledger_small.csv"
df.to_csv(filename, index=False)

# ── SUMMARY ─────────────────────────────────────────────────────────────────
print(f"\n✅  Saved {len(df):,} transactions to '{filename}'")
print(f"\n📊  EXPECTED DASHBOARD VALUES")
unique_nodes = pd.concat([df['Sender_ID'], df['Receiver_Acct']]).nunique()
print(f"    ├─ Total Accounts Scanned : ~{unique_nodes:,}")
print(f"    ├─ Transactions Processed : {len(df):,}")
print(f"    ├─ Known Malicious Hubs   : {total_fraudsters}  (nodes named FRAUD_ACT_*)")
print(f"    └─ Newly Identified Mules : {total_mule_nodes}   (MULE_HUB_* + OFFSHORE_* + SHELL_*)")
print(f"\n    Upload '{filename}' to the dashboard and verify these numbers match.")
