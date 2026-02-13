"""
MLOps Portfolio - API de prédiction avec monitoring
Auteur: [Ton Prénom Nom]
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from typing import Optional
import logging
import os

from app.monitoring import DriftDetector, MetricsLogger
from model.train import load_or_train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MLOps Portfolio API",
    description="Pipeline MLOps complet : entraînement, déploiement, monitoring de drift",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement du modèle au démarrage
model = None
drift_detector = DriftDetector()
metrics_logger = MetricsLogger()

@app.on_event("startup")
async def startup_event():
    global model
    logger.info("Chargement du modèle...")
    model = load_or_train_model()
    logger.info("Modèle prêt ✓")


# --- Schémas Pydantic ---

class PredictionInput(BaseModel):
    features: list[float]
    request_id: Optional[str] = None

class PredictionOutput(BaseModel):
    prediction: float
    confidence: float
    model_version: str
    latency_ms: float
    drift_alert: bool
    timestamp: str

class TrainRequest(BaseModel):
    n_samples: int = 1000
    experiment_name: str = "portfolio-experiment"


# --- Endpoints ---

@app.get("/")
def root():
    return {
        "message": "MLOps Portfolio API 🚀",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput, background_tasks: BackgroundTasks):
    """Endpoint de prédiction avec détection de drift automatique."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    start = time.time()
    features = np.array(input_data.features).reshape(1, -1)

    # Prédiction
    prediction = float(model.predict(features)[0])
    
    # Confiance (probabilité si classifieur, sinon score normalisé)
    try:
        proba = model.predict_proba(features)[0]
        confidence = float(max(proba))
    except AttributeError:
        confidence = 0.95

    latency = (time.time() - start) * 1000

    # Détection de drift en arrière-plan
    drift_alert = drift_detector.check(input_data.features)
    background_tasks.add_task(metrics_logger.log, prediction, latency, drift_alert)

    return PredictionOutput(
        prediction=prediction,
        confidence=confidence,
        model_version=os.getenv("MODEL_VERSION", "1.0.0"),
        latency_ms=round(latency, 2),
        drift_alert=drift_alert,
        timestamp=datetime.now().isoformat()
    )

@app.post("/train")
def train_model(request: TrainRequest):
    """Lance un entraînement et log les métriques dans MLflow."""
    from model.train import train_and_log
    
    result = train_and_log(
        n_samples=request.n_samples,
        experiment_name=request.experiment_name
    )
    return {"status": "success", "metrics": result}

@app.get("/metrics")
def get_metrics():
    """Retourne les métriques de monitoring en temps réel."""
    return metrics_logger.get_summary()

@app.get("/drift")
def get_drift_status():
    """Statut du drift sur les dernières requêtes."""
    return drift_detector.get_report()
