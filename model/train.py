"""
Entraînement du modèle avec tracking MLflow complet.
Dataset : classification de données synthétiques (remplaçable par tes vraies données).
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from sklearn.datasets import make_classification
import joblib
import os
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

MODEL_PATH = "model/saved_model.pkl"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")


def generate_dataset(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Génère un dataset de classification.
    À remplacer par tes vraies données dans un vrai projet !
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        weights=[0.6, 0.4],  # légèrement imbalancé, plus réaliste
        random_state=42
    )
    return X, y


def build_pipeline(model_type: str = "gradient_boosting") -> Pipeline:
    """Construit le pipeline sklearn : preprocessing + modèle."""
    models = {
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42
        ),
        "logistic": LogisticRegression(max_iter=1000, random_state=42)
    }

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", models.get(model_type, models["gradient_boosting"]))
    ])


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Calcule toutes les métriques d'évaluation."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }


def train_and_log(
    n_samples: int = 1000,
    model_type: str = "gradient_boosting",
    experiment_name: str = "portfolio-experiment"
) -> Dict:
    """
    Entraîne un modèle et log TOUT dans MLflow :
    - Paramètres
    - Métriques
    - Le modèle sérialisé
    - Les métadonnées
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    X, y = generate_dataset(n_samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

        # --- Log des paramètres ---
        mlflow.log_params({
            "model_type": model_type,
            "n_samples": n_samples,
            "n_features": X.shape[1],
            "test_size": 0.2,
            "train_size": len(X_train),
        })

        # --- Entraînement ---
        pipeline = build_pipeline(model_type)
        pipeline.fit(X_train, y_train)

        # --- Évaluation ---
        metrics = evaluate_model(pipeline, X_test, y_test)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
        metrics["cv_f1_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_f1_std"] = round(cv_scores.std(), 4)

        # --- Log des métriques ---
        mlflow.log_metrics(metrics)

        # --- Log du modèle ---
        mlflow.sklearn.log_model(pipeline, "model")

        # --- Sauvegarde locale ---
        os.makedirs("model", exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)

        logger.info(f"✅ Modèle entraîné — Accuracy: {metrics['accuracy']} | F1: {metrics['f1_score']} | AUC: {metrics['roc_auc']}")
        logger.info(f"📊 Run MLflow loggé dans {MLFLOW_TRACKING_URI}")

        return metrics


def load_or_train_model():
    """Charge le modèle depuis le disque, ou l'entraîne si absent."""
    if os.path.exists(MODEL_PATH):
        logger.info(f"Modèle chargé depuis {MODEL_PATH}")
        return joblib.load(MODEL_PATH)
    else:
        logger.info("Aucun modèle trouvé, entraînement en cours...")
        train_and_log()
        return joblib.load(MODEL_PATH)
