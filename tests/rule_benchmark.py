"""
tests/rule_benchmark.py
-----------------------
Scores the transaction rules against synthetic ledgers with known ground truth.

  python -m tests.rule_benchmark [extra_ledger.csv ...]

Datasets
  sample   dashboard/frontend/public/sample_ledger.csv  (truth from account names)
  neutral  generated here: neutral account names (ACC_...), so nothing can be
           detected by name, plus hard negatives — shops, landlords that collect
           from many tenants, employers that pay many staff
"""

import contextlib
import io
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from api.main import _analyze_transactions, _warm_up  # noqa: E402


def truth_from_names(accounts) -> dict:
    out = {}
    for a in accounts:
        u = str(a).upper()
        out[a] = "fraud" if "FRAUD" in u else "mule" if any(w in u for w in ("MULE", "OFFSHORE", "SHELL")) else "normal"
    return out


def make_neutral_ledger(seed=7, messy=False):
    """Rings: victims -> fraud actors -> hub -> shell -> shell -> offshore. Names carry no hints.

    messy=True adds what real data has and the clean version lacks:
      * fraud actors also spend small amounts at 0-3 shops
      * every third ring has no fraud-actor layer: victims pay the hub directly
      * franchise outlets (legitimate): collect from customers who also shop
        elsewhere, then forward takings to one owner account
    """
    rng = np.random.default_rng(seed)
    ids = iter(rng.permutation(200_000))
    new = lambda: f"ACC_{next(ids):06d}"          # noqa: E731
    users     = [new() for _ in range(4000)]
    shops     = [new() for _ in range(300)]
    landlords = [new() for _ in range(15)]        # hard negative: many in, few out
    employers = [new() for _ in range(10)]        # hard negative: few in, many out
    truth = {a: "normal" for a in users + shops + landlords + employers}
    rows = []
    pay = lambda s, r, lo, hi: rows.append((s, r, round(float(rng.uniform(lo, hi)), 2)))   # noqa: E731

    for _ in range(30_000):
        s = users[rng.integers(len(users))]
        pay(s, shops[rng.integers(len(shops))] if rng.random() > 0.2 else users[rng.integers(len(users))], 5, 1500)
    for ll in landlords:
        for t in rng.choice(users, rng.integers(20, 60), replace=False):
            pay(t, ll, 400, 1500)
        for _ in range(rng.integers(1, 3)):
            pay(ll, shops[rng.integers(len(shops))], 200, 5000)
    for e in employers:
        for w in rng.choice(users, rng.integers(30, 80), replace=False):
            pay(e, w, 1500, 4000)

    if messy:
        for _ in range(8):
            outlet, owner = new(), new()
            truth.update({outlet: "normal", owner: "normal"})
            takings = 0.0
            for c in rng.choice(users, rng.integers(30, 70), replace=False):
                amt = float(rng.uniform(20, 300)); takings += amt
                rows.append((c, outlet, round(amt, 2)))
            rows.append((outlet, owner, round(takings * 0.8, 2)))

    for ring in range(6):
        hub, shell1, shell2, offshore = new(), new(), new(), new()
        actors = [new() for _ in range(rng.integers(30, 90))]
        truth.update({hub: "mule", shell1: "mule", shell2: "mule", offshore: "mule"})
        total = 0.0
        if messy and ring % 3 == 2:                       # no actor layer: victims pay the hub directly
            for v in rng.choice(users, rng.integers(30, 90), replace=False):
                amt = float(rng.uniform(300, 3000)); total += amt
                rows.append((v, hub, round(amt, 2)))
            actors = []
        for a in actors:
            truth[a] = "fraud"
            got = 0.0
            for v in rng.choice(users, rng.integers(3, 12), replace=False):
                amt = float(rng.uniform(300, 3000)); got += amt
                rows.append((v, a, round(amt, 2)))
            if messy:
                for _ in range(rng.integers(0, 4)):       # personal spending
                    pay(a, shops[rng.integers(len(shops))], 10, 150)
            rows.append((a, hub, round(got * 0.95, 2))); total += got * 0.95
        rows += [(hub, shell1, round(total * 0.97, 2)), (shell1, shell2, round(total * 0.94, 2)),
                 (shell2, offshore, round(total * 0.9, 2))]

    df = pd.DataFrame(rows, columns=["payer_account", "payee_account", "txn_value"]).sample(frac=1, random_state=seed)
    return df.reset_index(drop=True), truth


def score(name, df, sender, receiver, amount, truth) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):          # silence the analysis logs
        _, _, _, ev = _analyze_transactions(df, sender, receiver, None, amount)
    accounts = list(ev["nodes"])
    t = np.array([truth.get(a, "normal") for a in accounts])
    p = np.asarray(ev["groups"], dtype=str)
    bad, flagged = t != "normal", p != "normal"
    tp = int((bad & flagged).sum())
    precision = tp / max(int(flagged.sum()), 1)
    recall    = tp / max(int(bad.sum()), 1)
    role_ok   = int(((t == p) & bad).sum())
    print(f"{name:<9} accounts {len(accounts):>6,} | truly bad {int(bad.sum()):>4} | flagged {int(flagged.sum()):>5} | "
          f"precision {precision:6.1%} | recall {recall:6.1%} | correct role {role_ok}/{int(bad.sum())}")
    return {"precision": precision, "recall": recall, "role_ok": role_ok, "bad": int(bad.sum()), "flagged": int(flagged.sum())}


def named_ledger(path):
    df = pd.read_csv(path).head(100_000)
    truth = truth_from_names(pd.unique(df[["Sender_ID", "Receiver_Acct"]].values.ravel()))
    return df, "Sender_ID", "Receiver_Acct", "Tx_Amount", truth


def main(extra_ledgers=()) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        _warm_up()                       # normally done at server start; the analysis waits for it
    neutral, neutral_truth = make_neutral_ledger()
    jobs = {"sample": named_ledger("dashboard/frontend/public/sample_ledger.csv"),
            "neutral": (neutral, "payer_account", "payee_account", "txn_value", neutral_truth)}
    messy, messy_truth = make_neutral_ledger(seed=11, messy=True)
    jobs["messy"] = (messy, "payer_account", "payee_account", "txn_value", messy_truth)
    for path in extra_ledgers:
        jobs[path.replace("\\", "/").split("/")[-1][:9]] = named_ledger(path)
    return {name: score(name, *job) for name, job in jobs.items()}


if __name__ == "__main__":
    main(sys.argv[1:])
