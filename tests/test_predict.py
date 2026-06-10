# Один валидный заказ, от него отталкиваются остальные тесты (подменяя по одному полю)
VALID_PAYLOAD = {
    "price": 120.0,
    "freight": 20.0,
    "installments": 1,
    "n_items": 1,
    "customer_state": "SP",
    "promised_lead_time_days": 15,
    "purchase_month": 6,
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_happy_path(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["negative_review_probability"] <= 1.0
    assert isinstance(body["is_high_risk"], bool)
    assert body["threshold"] > 0


def test_predict_negative_price_returns_422(client):
    payload = {**VALID_PAYLOAD, "price": -5.0}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_unknown_state_returns_422(client):
    payload = {**VALID_PAYLOAD, "customer_state": "ZZ"}
    assert client.post("/predict", json=payload).status_code == 422


def test_gradio_root_renders(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "gradio" in r.text.lower()