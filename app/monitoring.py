"""
Monitoring : Détection de drift + logging des métriques
"""

import numpy as np
from collections import deque
from datetime import datetime
from typing import List, Dict
import statistics
import logging

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Détecte le data drift en comparant la distribution
    des requêtes récentes vs les données d'entraînement.
    Méthode : comparaison de moyenne + écart-type (simple, robuste).
    """

    def __init__(self, window_size: int = 100, threshold: float = 2.5):
        self.window_size = window_size
        self.threshold = threshold  # nb d'écarts-types pour déclencher l'alerte
        self.recent_values: deque = deque(maxlen=window_size)
        self.reference_mean: float = 0.0
        self.reference_std: float = 1.0
        self.drift_count: int = 0
        self.total_requests: int = 0

        # Valeurs de référence simulées (remplace par tes vraies données d'entraînement)
        np.random.seed(42)
        reference_data = np.random.normal(0, 1, 500).tolist()
        self.reference_mean = statistics.mean(reference_data)
        self.reference_std = statistics.stdev(reference_data)

    def check(self, features: List[float]) -> bool:
        """Retourne True si drift détecté."""
        self.total_requests += 1
        feature_mean = statistics.mean(features)
        self.recent_values.append(feature_mean)

        if len(self.recent_values) < 20:
            return False

        current_mean = statistics.mean(self.recent_values)
        z_score = abs(current_mean - self.reference_mean) / (self.reference_std + 1e-8)

        drift_detected = z_score > self.threshold
        if drift_detected:
            self.drift_count += 1
            logger.warning(f"⚠️  Drift détecté ! Z-score: {z_score:.2f}")

        return drift_detected

    def get_report(self) -> Dict:
        current_mean = statistics.mean(self.recent_values) if self.recent_values else 0
        return {
            "drift_count": self.drift_count,
            "total_requests": self.total_requests,
            "drift_rate_pct": round(self.drift_count / max(self.total_requests, 1) * 100, 2),
            "current_window_mean": round(current_mean, 4),
            "reference_mean": round(self.reference_mean, 4),
            "reference_std": round(self.reference_std, 4),
            "window_size": len(self.recent_values),
            "status": "drift_detected" if self.drift_count > 0 else "stable"
        }


class MetricsLogger:
    """
    Logger léger pour les métriques de production.
    Stocke en mémoire (remplaçable par PostgreSQL/Redis).
    """

    def __init__(self):
        self.predictions: List[float] = []
        self.latencies: List[float] = []
        self.drift_alerts: List[bool] = []
        self.timestamps: List[str] = []
        self.request_count: int = 0

    def log(self, prediction: float, latency_ms: float, drift_alert: bool):
        self.request_count += 1
        self.predictions.append(prediction)
        self.latencies.append(latency_ms)
        self.drift_alerts.append(drift_alert)
        self.timestamps.append(datetime.now().isoformat())

        # Garde seulement les 1000 derniers
        if len(self.predictions) > 1000:
            self.predictions.pop(0)
            self.latencies.pop(0)
            self.drift_alerts.pop(0)
            self.timestamps.pop(0)

    def get_summary(self) -> Dict:
        if not self.latencies:
            return {"status": "no_data", "request_count": 0}

        return {
            "request_count": self.request_count,
            "latency_ms": {
                "mean": round(statistics.mean(self.latencies), 2),
                "p50": round(sorted(self.latencies)[len(self.latencies) // 2], 2),
                "p95": round(sorted(self.latencies)[int(len(self.latencies) * 0.95)], 2),
                "max": round(max(self.latencies), 2),
            },
            "predictions": {
                "mean": round(statistics.mean(self.predictions), 4),
                "std": round(statistics.stdev(self.predictions), 4) if len(self.predictions) > 1 else 0,
                "unique_classes": list(set([round(p) for p in self.predictions]))
            },
            "drift_alerts_total": sum(self.drift_alerts),
            "drift_alert_rate_pct": round(sum(self.drift_alerts) / len(self.drift_alerts) * 100, 2),
            "last_updated": self.timestamps[-1] if self.timestamps else None
        }
