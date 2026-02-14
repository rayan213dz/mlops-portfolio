# 🚀 MLOps Portfolio — Pipeline ML End-to-End

> Projet démontrant une maîtrise complète du cycle de vie d'un modèle ML en production :
> entraînement, déploiement, monitoring de drift et CI/CD automatisé.

![CI/CD](https://github.com/TON_USERNAME/mlops-portfolio/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![MLflow](https://img.shields.io/badge/MLflow-2.10-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

## 🎯 Objectif

Ce projet implémente un système MLOps complet, couvrant :
- **Entraînement** : pipeline sklearn avec tracking MLflow
- **Déploiement** : API REST FastAPI containerisée avec Docker
- **Monitoring** : détection de data drift en temps réel (z-score)
- **CI/CD** : GitHub Actions — tests → validation modèle → deploy

## 🏗️ Architecture

```
mlops-portfolio/
├── app/
│   ├── main.py          # API FastAPI (predict, train, metrics, drift)
│   └── monitoring.py    # DriftDetector + MetricsLogger
├── model/
│   └── train.py         # Pipeline sklearn + MLflow tracking
├── tests/
│   └── test_all.py      # Tests unitaires & intégration (pytest)
├── .github/
│   └── workflows/
│       └── ci.yml       # Pipeline CI/CD GitHub Actions
├── Dockerfile
└── requirements.txt
```

## ⚡ Lancement rapide

```bash
# 1. Clone & install
git clone https://github.com/TON_USERNAME/mlops-portfolio
cd mlops-portfolio
pip install -r requirements.txt

# 2. Lancer l'API
uvicorn app.main:app --reload

# 3. Tester l'API
curl http://localhost:8000/docs   # Swagger UI interactif

# 4. Lancer les tests
pytest tests/ -v

# 5. Voir les runs MLflow
mlflow ui
```

## 🐳 Docker

```bash
docker build -t mlops-portfolio .
docker run -p 8000:8000 mlops-portfolio
```

## 📡 Endpoints API

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Info API |
| GET | `/health` | Santé du service |
| POST | `/predict` | Prédiction + détection drift |
| POST | `/train` | Lance un entraînement + log MLflow |
| GET | `/metrics` | Métriques de production (latence, drift) |
| GET | `/drift` | Rapport de drift détaillé |

### Exemple de prédiction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, -0.5, 1.2, 0.3, -0.8, 0.6, -0.2, 0.9, -0.4, 0.7]}'
```

```json
{
  "prediction": 1.0,
  "confidence": 0.87,
  "model_version": "1.0.0",
  "latency_ms": 3.42,
  "drift_alert": false,
  "timestamp": "2025-02-13T14:32:01"
}
```

## 🔍 Détection de Drift

Le système compare la distribution des requêtes en temps réel avec les données d'entraînement via un **z-score**. Si la moyenne glissante s'écarte de plus de 2.5 écarts-types de la référence, une alerte est déclenchée automatiquement.

```
GET /drift → { "drift_count": 3, "drift_rate_pct": 1.2, "status": "stable" }
```

## 📊 MLflow Tracking

Chaque entraînement logue automatiquement :
- Paramètres du modèle
- Métriques : Accuracy, Precision, Recall, F1, AUC-ROC, CV scores
- Le modèle sérialisé

```bash
mlflow ui  # → http://localhost:5000
```

## 🧪 Tests

```bash
pytest tests/ -v --cov=app --cov=model
```

Couverture : DriftDetector, MetricsLogger, Pipeline ML, Endpoints API.

## 🛠️ Stack Technique

| Composant | Technologie |
|-----------|-------------|
| API | FastAPI + Uvicorn |
| ML | Scikit-learn (GBM, RF, LR) |
| Tracking | MLflow |
| Sérialisation | Joblib |
| Tests | Pytest |
| CI/CD | GitHub Actions |
| Container | Docker |
| Deploy | Railway / Render |

## 👤 Auteur

**Rayan MALKI** — Étudiant M1 Data & IA
- GitHub: [@Rayanmlk](https://github.com/Rayanmlk)
- LinkedIn: [rayan malki](https://www.linkedin.com/in/rayan-malki/)
