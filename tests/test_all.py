"""
Tests unitaires et d'intégration — MLOps Portfolio
Lance avec : pytest tests/ -v
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.monitoring import DriftDetector, MetricsLogger
from model.train import generate_dataset, build_pipeline, evaluate_model, train_and_log


# ─── Tests Monitoring ───────────────────────────────────────────────

class TestDriftDetector:

    def test_no_drift_on_normal_data(self):
        detector = DriftDetector()
        normal_features = [0.1, -0.2, 0.3, -0.1, 0.0]
        for _ in range(25):
            result = detector.check(normal_features)
        assert result == False

    def test_drift_on_anomalous_data(self):
        detector = DriftDetector(threshold=2.0)
        anomalous = [100.0, 95.0, 88.0, 102.0, 97.0]
        results = [detector.check(anomalous) for _ in range(30)]
        assert any(results)

    def test_report_structure(self):
        detector = DriftDetector()
        report = detector.get_report()
        for key in ["drift_count", "total_requests", "drift_rate_pct", "status"]:
            assert key in report

    def test_window_size_respected(self):
        detector = DriftDetector(window_size=10)
        for i in range(50):
            detector.check([float(i)] * 5)
        assert len(detector.recent_values) <= 10


class TestMetricsLogger:

    def test_log_and_retrieve(self):
        ml = MetricsLogger()
        ml.log(1.0, 42.5, False)
        ml.log(0.0, 38.1, True)
        summary = ml.get_summary()
        assert summary["request_count"] == 2
        assert summary["latency_ms"]["mean"] == pytest.approx(40.3, abs=0.1)

    def test_empty_logger(self):
        ml = MetricsLogger()
        summary = ml.get_summary()
        assert summary["request_count"] == 0

    def test_drift_rate(self):
        ml = MetricsLogger()
        ml.log(1.0, 10.0, True)
        ml.log(0.0, 10.0, False)
        summary = ml.get_summary()
        assert summary["drift_alert_rate_pct"] == 50.0


# ─── Tests Modèle ───────────────────────────────────────────────────

class TestModel:

    def test_dataset_generation(self):
        X, y = generate_dataset(n_samples=200)
        assert X.shape == (200, 10)
        assert y.shape == (200,)
        assert set(y) == {0, 1}

    def test_pipeline_predict(self):
        X, y = generate_dataset(500)
        pipeline = build_pipeline("gradient_boosting")
        pipeline.fit(X[:400], y[:400])
        preds = pipeline.predict(X[400:])
        assert len(preds) == 100
        assert all(p in [0, 1] for p in preds)

    def test_pipeline_proba(self):
        X, y = generate_dataset(300)
        pipeline = build_pipeline("random_forest")
        pipeline.fit(X[:250], y[:250])
        probas = pipeline.predict_proba(X[250:])
        assert probas.shape[1] == 2
        assert all(abs(p[0] + p[1] - 1.0) < 1e-6 for p in probas)

    def test_metrics_all_present(self):
        X, y = generate_dataset(300)
        pipeline = build_pipeline("logistic")
        pipeline.fit(X[:250], y[:250])
        metrics = evaluate_model(pipeline, X[250:], y[250:])
        for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0

    def test_accuracy_above_threshold(self):
        X, y = generate_dataset(1000)
        pipeline = build_pipeline("gradient_boosting")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        assert metrics["accuracy"] > 0.55


# ─── Tests API (intégration) ─────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    Client de test avec lifespan activé (charge le modèle comme en prod).
    scope="module" = un seul client pour tous les tests API, plus rapide.
    """
    from app.main import app
    with TestClient(app) as c:
        yield c


class TestAPI:

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_root_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_predict_valid_input(self, client):
        # 19 features = colonnes du dataset Telco Churn après encodage
        # (gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService,
        #  MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
        #  DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
        #  Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges)
        features_telco = [1, 0, 1, 0, 12, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 2, 65.5, 780.0]
        response = client.post("/predict", json={"features": features_telco})
        assert response.status_code == 200, f"Erreur: {response.json()}"
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "latency_ms" in data
        assert "drift_alert" in data

    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_drift_endpoint(self, client):
        response = client.get("/drift")
        assert response.status_code == 200
