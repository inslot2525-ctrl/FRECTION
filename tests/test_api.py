"""End-to-end API tests: upload, verdicts, search, explanations."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import main

SAMPLE = "dashboard/frontend/public/sample_ledger.csv"


@pytest.fixture(scope="module")
def client():
    with TestClient(main.app) as c:
        yield c


@pytest.fixture(scope="module")
def sample(client):
    with open(SAMPLE, "rb") as f:
        return client.post("/api/analyze", files={"file": ("sample_ledger.csv", f, "text/csv")}).json()


def explain(client, analysis, account):
    return client.get(f"/api/analysis/{analysis['analysis_id']}/explain", params={"account": account}).json()


def test_health(client):
    assert client.get("/api/health").json()["status"] == "Frection API online"


def test_sample_counts_match_the_generator(sample):
    # generate_test_data.py plants 3 rings: 150 fraud actors and 9 mule accounts
    assert sample["mode"] == "transactions"
    assert sample["metrics"]["known_fraudsters"] == 150
    assert sample["metrics"]["suspected_mules"] == 9


def test_hub_is_explained_as_a_collection_hub(client, sample):
    result = explain(client, sample, "MULE_HUB_CRITICAL_0")
    assert result["group"] == "mule"
    assert any(r["title"] == "Collection hub" for r in result["reasons"])


def test_layering_follows_the_money(client, sample):
    for account in ("SHELL_CORP_0", "OFFSHORE_CAYMAN_0"):
        assert explain(client, sample, account)["group"] == "mule"


def test_shops_and_their_customers_are_not_flagged(client, sample):
    # the old fan-in rule flagged every shop, and then everyone who paid one
    assert explain(client, sample, "M_113")["group"] == "normal"
    assert explain(client, sample, "U_972")["group"] == "normal"


def test_search_and_unknown_account(client, sample):
    found = client.get(f"/api/analysis/{sample['analysis_id']}/accounts", params={"q": "mule_hub"}).json()
    assert found["total"] == 3
    missing = client.get(f"/api/analysis/{sample['analysis_id']}/explain", params={"account": "nope"})
    assert missing.status_code == 404


def test_neighbourhood_adds_accounts_outside_the_drawing(client, sample):
    nb = client.get(f"/api/analysis/{sample['analysis_id']}/neighbourhood", params={"account": "SHELL_CORP_0"}).json()
    assert {n["id"] for n in nb["nodes"]} == {"SHELL_CORP_0", "MULE_HUB_CRITICAL_0", "OFFSHORE_CAYMAN_0"}


def test_customer_table_uses_records_mode(client):
    rng = np.random.default_rng(1)
    rows = ["CustomerID,income,age,Class"]
    for i in range(300):
        fraud = i % 10 == 0
        rows.append(f"{50_000 + i},{rng.normal(90_000 if fraud else 30_000, 5_000):.0f},{rng.integers(20, 70)},{int(fraud)}")
    r = client.post("/api/analyze", files={"file": ("customers.csv", "\n".join(rows).encode(), "text/csv")}).json()
    assert r["mode"] == "entities"
    assert r["column_mapping"]["id"] == "CustomerID" and r["column_mapping"]["fraud"] == "Class"
    assert r["metrics"]["total_nodes"] == 300 and r["metrics"]["known_fraudsters"] == 30


@pytest.mark.parametrize("name,body,status", [
    ("notes.txt", b"hello", 400),
    ("empty.csv", b"", 400),
    ("one_col.csv", b"a\n1\n2\n", 400),
])
def test_bad_uploads_are_rejected_cleanly(client, name, body, status):
    assert client.post("/api/analyze", files={"file": (name, body, "text/csv")}).status_code == status


def test_user_can_override_columns(client):
    csv = b"a,b,c\nX1,H,5\nX2,H,5\nX3,H,5\nH,Z,14\n"
    r = client.post("/api/analyze", files={"file": ("t.csv", csv, "text/csv")},
                    data={"mode": "transactions", "sender_col": "a", "receiver_col": "b", "amount_col": "c"}).json()
    assert r["mode"] == "transactions" and r["column_mapping"]["sender"] == "a"
