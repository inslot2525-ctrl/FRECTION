"""Column understanding: the part that decides what an uploaded file means."""

import numpy as np
import pandas as pd

from api.ingest import binary_label, detect_columns, normalise_frame, read_upload, tokens


def frame(csv: str) -> pd.DataFrame:
    return normalise_frame(read_upload(csv.encode("utf-8"), 100_000))


def test_tokens_split_camel_case_and_separators():
    assert tokens("CustomerID") == {"customer", "id"}
    assert tokens("acct_from") == {"acct", "from"}


def test_short_patterns_need_a_whole_word():
    # "to" must not match inside "CustomerID": this file has no receiver column
    df = frame("CustomerID,A1,A2,Class\n" + "\n".join(f"{1000 + i},{i % 7},{i * 1.5},{i % 2}" for i in range(40)))
    mapping = detect_columns(df)
    assert mapping["mode"] == "entities"
    assert mapping["id"] == "CustomerID"
    assert mapping["fraud"] == "Class"


def test_named_transaction_columns():
    df = frame("nameOrig,nameDest,amount,isFraud\nC1,C2,10,0\nC2,C3,5,1\n")
    mapping = detect_columns(df)
    assert (mapping["mode"], mapping["sender"], mapping["receiver"]) == ("transactions", "nameOrig", "nameDest")
    assert mapping["amount"] == "amount" and mapping["fraud"] == "isFraud"


def test_meaningless_names_are_paired_by_value_overlap():
    rng = np.random.default_rng(0)
    accounts = [f"acc_{i}" for i in range(200)]
    df = pd.DataFrame({"col_x": rng.choice(accounts, 1000), "col_y": rng.choice(accounts, 1000),
                       "v": rng.uniform(1, 99, 1000).round(2)})
    mapping = detect_columns(df)
    assert mapping["mode"] == "transactions"
    assert {mapping["sender"], mapping["receiver"]} == {"col_x", "col_y"}


def test_semicolons_spaced_headers_and_blank_rows():
    df = frame(" Account No ; Balance ; Fraud? \nCU-1;€1,200;Yes\n;;\nCU-2;€90;No\n")
    assert list(df.columns) == ["Account No", "Balance", "Fraud?"]
    assert len(df) == 2


def test_money_text_is_not_an_account_column():
    rows = "\n".join(f"CU-{i:03d};€{1000 + i * 37:,};{'DE' if i % 2 else 'FR'};{'Yes' if i % 9 == 0 else 'No'}" for i in range(60))
    mapping = detect_columns(frame("Account No;Balance;Country;Fraud?\n" + rows))
    assert mapping["mode"] == "entities" and mapping["id"] == "Account No"


def test_binary_label_variants():
    assert binary_label(pd.Series([1, 0, 1])).tolist() == [True, False, True]
    assert binary_label(pd.Series(["Yes", "No", "yes"])).tolist() == [True, False, True]
    assert binary_label(pd.Series(["fraud", "legit"])).tolist() == [True, False]
    # two-valued codes without a 1: the rarer value is the positive class
    assert binary_label(pd.Series([2, 2, 2, 4])).tolist() == [False, False, False, True]
