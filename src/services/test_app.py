from fastapi.testclient import TestClient
from src.services.app import app

client = TestClient(app)

def test_health_endpoint():
    """Verify that the health check endpoint returns 200 and indicates model readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}

def test_predict_endpoint_high_risk():
    """Verify that high-friction inputs return a valid risk prediction payload."""
    payload = {
        "balance_to_max_ratio": 0.2,
        "tx_velocity_drop_ratio": 0.8,
        "failed_tx_count_7d": 4,
        "support_tickets_30d": 3,
        "is_active_credit_card_user": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "risk_score" in data
    assert "risk_status" in data
    assert "downstream_routing_action" in data