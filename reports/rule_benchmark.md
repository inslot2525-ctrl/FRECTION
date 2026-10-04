# Transaction rules — before and after

Scored against synthetic ledgers with known ground truth
(`python -m tests.rule_benchmark`). "Flagged" means labelled fraud or mule.

| Dataset | Accounts | Truly bad | | Flagged | Precision | Recall |
|---|---|---|---|---|---|---|
| sample (bundled demo) | 1,315 | 159 | before | 1,030 | 15.4 % | 100 % |
| | | | **after** | 159 | **100 %** | 100 % |
| large ledger | 5,743 | 243 | before | 5,742 | 4.2 % | 100 % |
| | | | **after** | 243 | **100 %** | 100 % |
| neutral names + shops, landlords, employers | 4,701 | 376 | before | 4,671 | 7.8 % | 96.8 % |
| | | | **after** | 376 | **100 %** | 100 % |
| messy (see below) | 4,563 | 222 | **after** | 214 | **100 %** | 96.4 % |

**Before:** name matching (`FRAUD`, `MULE`, `SHELL`, `OFFSHORE`), a fan-in rule
that flagged every popular receiver including shops, and a cascade that then
flagged everyone who paid them. On the larger ledgers this flagged nearly every
account, so the 100 % recall meant nothing.

**After:** no name matching. A hub must forward money and be fed mainly by
pass-through accounts; layering follows the money downstream; fraud actors are
the feeders of a confirmed hub.

**The messy set** has neutral account names, fraud actors who also spend at
shops, franchise outlets that legitimately collect and forward, and two rings
with no fraud-actor layer. The 8 missed accounts are exactly those two rings:
victims paying a hub directly look the same as customers paying a business.

**Read this with care.** These ledgers are synthetic and share the rules' own
idea of what a ring looks like. The table shows the rules do what they were
designed to do and no longer flag ordinary accounts. It is not evidence of
performance on real bank data.
