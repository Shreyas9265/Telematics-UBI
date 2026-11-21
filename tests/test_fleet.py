# tests/test_fleet.py
from fastapi.testclient import TestClient

from src import api
from src.fleet import incident_prob
from src.pricing import premium_from_score, risk_score

client = TestClient(api.app)


def _load_sample():
    r = client.post("/load_csv", json={"clear": True})
    assert r.status_code == 200 and r.json()["loaded"] > 0


def test_fleet_matches_per_driver_endpoints():
    _load_sample()
    body = client.get("/fleet", params={"base_premium": 1200}).json()
    assert body["fleet"]["drivers"] == len(body["drivers"]) > 0
    d = body["drivers"][0]
    # same rule as /driver/{id}/score and /price
    assert client.get(f"/driver/{d['driver_id']}/score").json()["risk_score"] == d["risk_score"]
    price = client.post("/price", json={"driver_id": d["driver_id"], "base_premium": 1200}).json()
    assert price["premium"] == d["premium"]
    # drivers are sorted riskiest first, and the histogram counts every driver
    scores = [x["risk_score"] for x in body["drivers"]]
    assert scores == sorted(scores, reverse=True)
    assert sum(b["drivers"] for b in body["fleet"]["score_histogram"]) == len(scores)


def test_premium_formula_is_clamped():
    assert premium_from_score(1000, 50) == 1000
    assert premium_from_score(1000, 100) == 1350
    assert premium_from_score(1000, 0) == 700  # never below 0.7x
    assert risk_score(incident_prob(0)) == 13
    assert risk_score(incident_prob(200)) == 100


def test_fleet_points_for_map():
    _load_sample()
    body = client.get("/fleet/points", params={"limit": 50}).json()
    assert body["count"] == 50
    assert {"lat", "lon", "speed_kph"} <= set(body["points"][0])
