import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["AGENTNET_DB"] = os.path.join(tempfile.mkdtemp(), "t.db")

from fastapi.testclient import TestClient  # noqa: E402

from agentnet import safety  # noqa: E402
from agentnet.db import init_db  # noqa: E402
from agentnet.main import app  # noqa: E402

API = "/api/v1"
SWIGGY = "Food and grocery delivery, restaurant partnerships, festive food offers, payment and card partnerships."
PHONEPE = "E-commerce promotions, food delivery festive discounts, card payment partnerships, UPI merchant cashback."
HDFC = "Credit card and debit card offers, EMI, merchant partnerships, festive spend rewards on shopping and food."
BAJAJ = "Motorcycles, scooters, electric vehicles, dealership network, test rides and vehicle launches."


@pytest.fixture()
def client(tmp_path):
    init_db(str(tmp_path / "db.sqlite"))
    safety.limiter.reset()
    with TestClient(app) as c:
        init_db(str(tmp_path / "db.sqlite"))
        yield c


@pytest.fixture()
def make_agent(client):
    def _make(name, manifesto, industry="Misc", **kw):
        r = client.post(f"{API}/agents/register", json={"company_name": name, "industry_category": industry,
                                                         "interest_manifesto": manifesto, **kw})
        assert r.status_code == 201, r.text
        d = r.json()
        client.post(f"{API}/admin/agents/{d['agent_id']}/verify", headers={"X-Admin-Key": "dev-admin-key"})
        d["h"] = {"X-API-Key": d["api_key"]}
        return d
    return _make
