"""
Entraînement du modèle avec tracking MLflow complet.
Dataset : Telco Customer Churn (chargé depuis GitHub/URL — pas besoin de CSV local)
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import joblib
import os
import logging
from typing import Dict, Tuple
from datetime import datetime
from io import StringIO

logger = logging.getLogger(__name__)

MODEL_PATH = "model/saved_model.pkl"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")

# URL publique du dataset Telco Churn (IBM Sample Data)
DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def load_telco_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge et prépare le dataset Telco Customer Churn.
    Objectif : prédire si un client va résilier son abonnement (Churn = Yes/No).
    """
    try:
        logger.info("Chargement du dataset Telco Churn...")
        df = pd.read_csv(DATASET_URL)
        logger.info(f"Dataset chargé : {df.shape[0]} clients, {df.shape[1]} colonnes")
    except Exception as e:
        logger.warning(f"URL inaccessible ({e}), génération de données synthétiques...")
        return generate_fallback_dataset()

    # --- Nettoyage ---
    # TotalCharges contient des espaces pour les nouveaux clients
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Supprime customerID (inutile pour la prédiction)
    df.drop(columns=["customerID"], inplace=True)

    # --- Encodage de la cible ---
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    y = df["Churn"].values

    # --- Encodage des features catégorielles ---
    df.drop(columns=["Churn"], inplace=True)

    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.values.astype(float)
    logger.info(f"Features : {X.shape[1]} | Churn rate : {y.mean():.1%}")
    return X, y


def generate_fallback_dataset(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Dataset de secours si l'URL est inaccessible (ex: CI sans réseau)."""
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=n_samples, n_features=10, n_informative=6,
        n_redundant=2, n_classes=2, weights=[0.74, 0.26], random_state=42
    )
    return X, y


# Garde generate_dataset comme alias pour les tests
def generate_dataset(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    return generate_fallback_dataset(n_samples)


def build_pipeline(model_type: str = "gradient_boosting") -> Pipeline:
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
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
    }


def train_and_log(
    n_samples: int = 1000,
    model_type: str = "gradient_boosting",
    experiment_name: str = "portfolio-experiment"
) -> Dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    # Essaie le vrai dataset, fallback si pas de réseau
    try:
        X, y = load_telco_dataset()
        dataset_name = "telco_churn"
    except Exception:
        X, y = generate_fallback_dataset(n_samples)
        dataset_name = "synthetic_fallback"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(
        run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        mlflow.log_params({
            "model_type":   model_type,
            "dataset":      dataset_name,
            "n_samples":    len(X),
            "n_features":   X.shape[1],
            "churn_rate":   round(y.mean(), 3),
            "test_size":    0.2,
        })

        pipeline = build_pipeline(model_type)
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
        metrics["cv_f1_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_f1_std"]  = round(cv_scores.std(), 4)

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model")

        os.makedirs("model", exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)

        logger.info(
            f"✅ [{dataset_name}] Accuracy: {metrics['accuracy']} "
            f"| F1: {metrics['f1_score']} | AUC: {metrics['roc_auc']}"
        )
        return metrics


def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        logger.info(f"Modèle chargé depuis {MODEL_PATH}")
        return joblib.load(MODEL_PATH)
    logger.info("Aucun modèle trouvé, entraînement en cours...")
    train_and_log()
    return joblib.load(MODEL_PATH)
