from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_liveness_endpoint():
    response = client.get("/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_readiness_endpoint():
    response = client.get("/ready")
    assert response.status_code == 200


def test_version_endpoint():
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "app_version" in data


def test_model_info_endpoint():
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()
    assert "champion_model" in data


def test_predict_endpoint():
    payload = {
        "season": "summer",
        "yr": "2012",
        "mnth": 6.0,
        "hr": 18,
        "holiday": "No",
        "weekday": 2,
        "workingday": "Working Day",
        "weather": "Clear",
        "temp": 0.65,
        "hum": 0.45,
        "windspeed": 0.15,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_demand" in data
    assert data["predicted_demand"] >= 0
    assert "estimated_revenue_usd" in data
    assert "disclaimer" in data


def test_predict_batch_endpoint():
    payload = {
        "inputs": [
            {
                "season": "summer",
                "yr": "2012",
                "mnth": 6.0,
                "hr": 18,
                "holiday": "No",
                "weekday": 2,
                "workingday": "Working Day",
                "weather": "Clear",
                "temp": 0.65,
                "hum": 0.45,
                "windspeed": 0.15,
            },
            {
                "season": "winter",
                "yr": "2011",
                "mnth": 1.0,
                "hr": 8,
                "holiday": "No",
                "weekday": 1,
                "workingday": "Working Day",
                "weather": "Mist",
                "temp": 0.25,
                "hum": 0.75,
                "windspeed": 0.25,
            },
        ]
    }

    response = client.post("/predict_batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_predictions"] == 2
    assert len(data["predictions"]) == 2
