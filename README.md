# Enterprise MLOps Pipeline: MNIST Digit Classifier 🚀

[![CI Pipeline](https://github.com/your-username/mnist-mlops-project/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/mnist-mlops-project/actions/workflows/ci.yml)
[![Container Security Scan](https://github.com/your-username/mnist-mlops-project/actions/workflows/security_scan.yml/badge.svg)](https://github.com/your-username/mnist-mlops-project/actions/workflows/security_scan.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is a fully automated, secure, and scalable **End-to-End MLOps** ecosystem for handwritten digit recognition (MNIST), built from scratch to an **Enterprise-grade** standard.

---

## 🏗️ System Architecture & Technology Stack

* **Machine Learning & Optimization:** TensorFlow/Keras CNN, ONNX Runtime (maximum speed and low latency).
* **Version Control & Data Management:** Git, DVC (Data Version Control) for model and dataset tracking.
* **Backend & Frontend:** FastAPI (High-performance REST API), Streamlit (Interactive Web UI).
* **Containerization & Orchestration:** Docker, Docker Compose, Kubernetes, Helm.
* **CI/CD & GitOps:** GitHub Actions (Pytest, Quality Gate, Trivy Security Scan), ArgoCD.
* **Pipeline Automation:** Prefect (E2E MLOps pipeline orchestration).
* **Observability & Monitoring:** Prometheus, Grafana, Evidently AI (Data/Model Drift), OpenTelemetry & Jaeger (Distributed Tracing). ---

## 📁 Project Structure

```text
mnist-mlops-project/
├── .devcontainer/ # GitHub Codespaces dev container config
├── .github/workflows/ # CI/CD pipelines & security scans
├── api/ # FastAPI services (Standard, ONNX, Traced)
├── helm/ # Kubernetes Helm Charts
├── k8s/ # Kubernetes & ArgoCD manifests
├── models/ # Trained models & ONNX artifacts
├── tests/ # Pytest automated test suite
├── prefect_orchestration.py # Prefect orchestration flow
├── retraining_pipeline.py # Automated retraining loop
├── validate_model_gate.py # Model Quality Gate validation
└── Dockerfile # Container definition
```
