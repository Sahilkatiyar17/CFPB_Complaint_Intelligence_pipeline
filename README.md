# 🏛️ CFPB Complaint Intelligence Pipeline

### End-to-End NLP & MLOps System for Automated Complaint Triage

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-Gunicorn-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=flat)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-AWS%20EKS-326CE5?style=flat&logo=kubernetes&logoColor=white)](https://kubernetes.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat)](LICENSE)

**A production-grade Machine Learning system for financial institutions.** Automatically classifies customer complaint categories and detects urgent complaints — using NLP, dual ML models, full experiment tracking, and an automated CI/CD pipeline to Kubernetes.

[Problem](#the-problem) · [Architecture](#architecture) · [ML Models](#ml-models) · [Quick Start](#quick-start) · [API Reference](#api-reference) · [Tech Stack](#tech-stack) · [Deployment](#deployment)

---

## The Problem

[#the-problem](#the-problem)

Financial institutions receive thousands of customer complaints every day. Manual complaint analysis is slow, error-prone, and difficult to prioritize at scale:

- **Slow processing time** — complaints sit in queues before anyone reads them
- **Human errors** — inconsistent categorization across reviewers
- **Delayed urgent response** — time-sensitive complaints get buried in volume
- **No automated decision support** — every complaint is triaged manually, every time

Most academic ML projects stop after training a model. This project goes further — it implements the **complete production lifecycle**: experimentation, model versioning, containerized deployment, and continuous integration, demonstrating how a research prototype becomes an enterprise-ready AI system.

---

## Architecture

[#architecture](#architecture)

```
                    CFPB Complaint Dataset
                             │
                             ▼
                    Data Cleaning Pipeline
                             │
                             ▼
               NLP Preprocessing Pipeline
   (Tokenization, Stopwords, Lemmatization, Embeddings)
                             │
               ┌─────────────┴─────────────┐
               ▼                           ▼
      Model A Training              Model B Training
 (Category Classification)      (Urgency Classification)
               │                           │
               ▼                           ▼
     Model Evaluation             Model Evaluation
               │                           │
               └─────────────┬─────────────┘
                             ▼
                    MLflow Experiment Tracking
                             │
                             ▼
                     Model Registry (MLflow)
                             │
                Manual Approval / Registration
                             │
──────────────────────────────────────────────────────────
                     PRODUCTION PIPELINE
──────────────────────────────────────────────────────────
                             │
                             ▼
                  Flask Prediction API
                             │
                             ▼
                  Fetch Registered Models
                             │
                             ▼
                CI/CD Pipeline (GitHub Actions)
                             │
                Unit Tests + API Validation
                             │
                             ▼
                     Docker Image Build
                             │
                             ▼
                   Push Image to AWS ECR
                             │
                             ▼
                 Kubernetes Deployment (EKS)
                             │
                             ▼
                Live Prediction Service (LoadBalancer)
```

### Data Flow

[#data-flow](#data-flow)

```
Complaint Narrative
        │
        ▼
   Text Cleaning
        │
        ▼
 Text Vectorization (Embedding model)
        │
        ▼
┌───────────────┬───────────────┐
▼               ▼
Model A         Model B
(Category)      (Urgency)
└───────────────┴───────────────┘
        │
        ▼
   Prediction API
        │
        ▼
  Response to User
```

---

## ML Models

[#ml-models](#ml-models)

### Model A — Complaint Category Prediction

[#model-a](#model-a)

| | |
|---|---|
| **Input** | Complaint narrative (free text) |
| **Output** | One of five categories |
| **Categories** | Credit Card · Bank Account · Loan · Mortgage · Money Transfer |

### Model B — Urgency Prediction

[#model-b](#model-b)

| | |
|---|---|
| **Input** | Complaint narrative (free text) |
| **Output** | Binary — Urgent / Not Urgent |

### Algorithms Evaluated

[#algorithms-evaluated](#algorithms-evaluated)

Multiple algorithms were experimentally evaluated for each task, with the best-performing model selected based on evaluation metrics:

- Logistic Regression
- Support Vector Classifier (SVC)
- XGBoost
- Multi-Layer Perceptron (MLP)

### Training Pipeline

[#training-pipeline](#training-pipeline)

```
Dataset → Data Cleaning → Text Preprocessing → Feature Engineering
   → Train Multiple Models → Hyperparameter Tuning → Model Evaluation
   → MLflow Logging → Model Registry
```

---

## Quick Start

[#quick-start](#quick-start)

### Prerequisites

[#prerequisites](#prerequisites)

- Python 3.12+
- Docker
- AWS CLI configured with appropriate IAM permissions
- `kubectl` and `eksctl` (for Kubernetes deployment)

### 1. Clone the Repository

[#1-clone-the-repository](#1-clone-the-repository)

```bash
git clone https://github.com/<your-username>/CFPB_Complaint_Intelligence_pipeline.git
cd CFPB_Complaint_Intelligence_pipeline
```

### 2. Install Dependencies

[#2-install-dependencies](#2-install-dependencies)

```bash
pip install --upgrade pip setuptools
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r flask_app/requirements.txt
```

### 3. Download NLTK Data

[#3-download-nltk-data](#3-download-nltk-data)

```bash
python -m nltk.downloader stopwords wordnet
```

### 4. Set Environment Variables

[#4-set-environment-variables](#4-set-environment-variables)

```bash
export CAPSTONE_TEST=<your-test-secret>
```

### 5. Run Locally

[#5-run-locally](#5-run-locally)

```bash
cd flask_app
python app.py
```

App runs at `http://localhost:5000`

### 6. Run with Docker

[#6-run-with-docker](#6-run-with-docker)

```bash
docker build -t cfpb-flask-app .
docker run -p 5000:5000 -e CAPSTONE_TEST=<your-test-secret> cfpb-flask-app
```

---

## API Reference

[#api-reference](#api-reference)

### Prediction

[#prediction](#prediction)

```
POST /predict
Body: { "complaint_text": "I was charged twice for the same transaction and no one is responding to my calls." }

Returns predicted complaint category and urgency classification.
```

### Health Check

[#health-check](#health-check)

```
GET /health

Returns service status.
```

---

## Tech Stack

[#tech-stack](#tech-stack)

| Layer | Technology |
|---|---|
| Programming | Python |
| Machine Learning | Scikit-learn, XGBoost |
| NLP | NLTK, TF-IDF Vectorizer |
| Experiment Tracking | MLflow |
| API | Flask + Gunicorn |
| CI/CD | GitHub Actions |
| Containerization | Docker |
| Orchestration | Kubernetes (AWS EKS) |
| Container Registry | AWS ECR |
| Version Control | Git |

---

## MLOps Pipeline

[#mlops-pipeline](#mlops-pipeline)

```
Code Update → GitHub → CI/CD Trigger → Run Tests → Validate Flask API
   → Build Docker Image → Push to ECR → Deploy to Kubernetes
   → Application Starts → Load Latest Registered Models → Ready for Predictions
```

The pipeline runs in three stages via GitHub Actions:

1. **`project-testing`** — installs dependencies and runs unit tests against the Flask API
2. **`docker-build-push`** — builds the Docker image and pushes it to AWS ECR
3. **`deploy-to-eks`** — updates the kubeconfig, applies Kubernetes secrets, and deploys to the EKS cluster

---

## Project Structure

[#project-structure](#project-structure)

```
CFPB_Complaint_Intelligence_pipeline/
├── flask_app/
│   ├── app.py                  ← Flask prediction API
│   └── requirements.txt
├── tests/
│   └── test_flask_app.py       ← Unit tests
├── .github/
│   └── workflows/
│       └── ci.yaml             ← CI/CD pipeline definition
├── Dockerfile
├── deployment.yaml             ← Kubernetes Deployment + Service
└── README.md
```

---

## Deployment

[#deployment](#deployment)

### AWS EKS

[#aws-eks](#aws-eks)

```bash
eksctl create cluster \
  --name=flask-app-cluster \
  --region=us-east-1 \
  --nodegroup-name=flask-app-nodes \
  --node-type=t3.large \
  --nodes=1 \
  --nodes-min=1 \
  --nodes-max=1 \
  --managed
```

### Apply Deployment

[#apply-deployment](#apply-deployment)

```bash
aws eks update-kubeconfig --region us-east-1 --name flask-app-cluster
kubectl apply -f deployment.yaml
```

### Get the Public URL

[#get-the-public-url](#get-the-public-url)

```bash
kubectl get svc flask-app-service
```

The `EXTERNAL-IP` column gives the public LoadBalancer URL for the live prediction service.

---

## Novelty of Project

[#novelty-of-project](#novelty-of-project)

This project extends beyond standard model development by implementing the complete production lifecycle:

- Dual Machine Learning models for two related business tasks
- Experiment tracking and model versioning using MLflow
- Automated CI/CD pipeline (test → build → deploy)
- Docker-based containerization
- Kubernetes deployment on AWS EKS

---

## Contributing

[#contributing](#contributing)

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'feat: add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

[#license](#license)

MIT License — see [LICENSE](LICENSE) for details.

---

Built as a Capstone Project demonstrating end-to-end NLP & MLOps practices.
