# Handwritten Digit Classifier with CNN (MNIST) - MLOps Production Pipeline

This project is a production-ready **MLOps** infrastructure built around the **MNIST** handwritten digit classification dataset. It encompasses the end-to-end lifecycle of deep learning models, leveraging both **PyTorch** and **TensorFlow** frameworks for training, model versioning, automated orchestration, continuous monitoring, and GitOps-driven deployment into a Kubernetes environment.

---

## 🏗️ Project Structure

```text
Handwritten-Digit-Classifier-with-CNN-MNIST/
├── .devcontainer/              # GitHub Codespaces configuration
│   └── devcontainer.json
├── .github/
│   └── workflows/              # CI/CD pipelines (Test, Security Scan)
│       ├── ci.yml
│       └── security_scan.yml
├── api/
│   ├── app.py                  # FastAPI backend (Multi-model support)
│   └── app_onnx.py             # ONNX Runtime optimized API
|   └── tracing_app.py
├── inference/                  # Framework-specific inference scripts
│   ├── predict_pytorch.py      # PyTorch model inference
│   └── predict_tensorflow.py   # TensorFlow model inference
├── helm/
│   └── mnist-api/              # Kubernetes Helm Chart
├── k8s/
│   ├── deployment.yaml         # Kubernetes manifests
│   └── argocd-app.yaml         # ArgoCD GitOps manifest
├── models/                     # Model artifacts (tracked with DVC)
│   ├── pytorch_cnn.pt          # PyTorch model weights
│   ├── tensorflow_cnn.keras    # TensorFlow model weights
│   └── mnist.onnx              # Unified ONNX model
├── notebook                    # Jupyter notebook (EDA & Training)
├── tests/
│   └── test_api.py             # Pytest unit tests
├── custom_images/              # Custom images for manual testing
├── plots/                      # Visualization charts and plots
├── dvc.yaml                    # DVC pipeline configuration
├── Dockerfile                  # Production Dockerfile
├── docker-compose.yml          # Local multi-service orchestration
├── locustfile.py               # Load testing script
├── monitor_drift.py            # Evidently AI drift monitoring
├── prefect_orchestration.py    # Prefect E2E workflow orchestration
├── retraining_pipeline.py      # Automated model retraining pipeline
├── validate_model_gate.py      # Quality Gate validation script
├── requirements.txt            # Python dependencies (PyTorch + TF)
└── README.md                   # Project documentation
```

---

## 🚀 Tech Stack

* **Deep Learning:** PyTorch, TensorFlow, Keras, ONNX Runtime
* **API & Serving:** FastAPI, Uvicorn, Docker, Docker Compose
* **MLOps & Orchestration:** DVC (Data Version Control), Prefect, Evidently AI
* **CI/CD & DevOps:** GitHub Actions, Helm Charts, ArgoCD, Kubernetes (k8s)
* **Testing:** Pytest (Unit Testing), Locust (Load Testing)

---

## 🛠️ Getting Started Locally

### 1. Clone the Repository and Navigate
```bash
git clone <repository-url>
cd Handwritten-Digit-Classifier-with-CNN-MNIST
```

### 2. Set Up a Virtual Environment and Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Pull Model Artifacts via DVC
Large model weights (`.pt`, `.keras`, `.onnx`) are tracked outside of Git using DVC:
```bash
dvc pull
```

### 4. Run the FastAPI Application
Start the multi-model supported production backend server locally:
```bash
uvicorn api.app:app --reload --port 8000
```
Open your browser and navigate to `http://localhost:8000/docs` to explore the interactive API documentation.

---

## 🔄 MLOps Pipeline & Automation

* **End-to-End Orchestration (`prefect_orchestration.py`):** Coordinates data fetching, preprocessing, training, and model conversion workflows.
* **Model Quality Gate (`validate_model_gate.py`):** Ensures newly trained models pass strictly defined accuracy thresholds before promotion.
* **Data & Concept Drift (`monitor_drift.py`):** Utilizes **Evidently AI** to detect performance drops or structural changes in real-world inference traffic.

### 🛡️ Key MLOps Components
* **Quality Gate:** If the new model's accuracy falls below the required threshold (98.5%), the CI/CD process automatically halts (validate_model_gate.py).

* **ONNX Optimization:** Inference speed has been increased by using the lightweight ONNX Runtime instead of the heavy TensorFlow library.

* **Security Scanning:** Each Docker image is scanned for critical security vulnerabilities using Trivy.

* **GitOps Deployment:** Cluster management is fully automated using ArgoCD and Helm.

* **Observability:** System metrics and distributed tracing are monitored in real-time using Prometheus, Grafana, and Jaeger.
---

## 🐳 Docker & Kubernetes Deployment

### Local Orchestration with Docker Compose
To build and spin up the complete API and supporting monitoring services simultaneously:
```bash
docker-compose up --build
```

### Production Deployment via GitOps (ArgoCD)
This project is fully packaged for Kubernetes environments using Helm. To spin up the CD pipeline using ArgoCD:
```bash
kubectl apply -f k8s/argocd-app.yaml
```

---

## 🧪 Testing

* **Unit Testing (Pytest):** Run `pytest tests/` to execute API routing and model inference integrity assertions.
* **Load Testing (Locust):** Evaluate backend behavior under high concurrency thresholds:
  ```bash
  locust -f locustfile.py
  ```



# IPYNB File - Handwritten Digit Classifier with a CNN (MNIST)

Building a Convolutional Neural Network (CNN) trained from scratch (random initialization) on the MNIST dataset, comparing it under identical conditions with a classic Multi-Layer Perceptron (MLP) baseline, and testing the model on entirely new, real-world handwritten digits.

## ✨ Project Features

* [x] **Data Pipeline:** The MNIST dataset is loaded using a standard loader, normalized, and explicitly split into 3 sets: Train / Validation / Test.
* [x] **CNN Architecture:** Designed from scratch, containing at least 2 Convolution+Pooling blocks followed by fully-connected (Dense) layers. (No pretrained weights were used).
* [x] **MLP Baseline Comparison:** A simple fully-connected network trained on the exact same data, splits, and number of epochs for a fair comparison.
* [x] **Training Curves:** Loss and accuracy plots per epoch for both the CNN and MLP.
* [x] **Confusion Matrix:** Computed on the test data. The most confused digit pair is identified along with a hypothesis for the cause.
* [x] **Custom Image Test:** The model is tested on 5 real-world photographed/drawn digit images not present in MNIST, with the image preprocessing (resize, grayscale, invert) shown step-by-step.
* [x] **Model Persistence:** Trained weights are saved to disk (e.g., `.onnx` or `.pt`) and successfully loaded in another script for inference without retraining.
* [x] **Data Augmentation:** Image rotation and shift are applied, and their impact on validation accuracy is demonstrated.
* [x] **Filter Visualization:** The learned feature filters of the first Convolutional (Conv1) layer are visualized.
* [x] **Regularization:** Dropout and Batch Normalization are added, and the results of regularized vs. unregularized models are compared.
* [x] **API Endpoint (FastAPI):** A small FastAPI/Flask server is set up that accepts an uploaded image and returns the predicted digit.

## 🏗️ CNN Model Architecture

The model consists of the following layers and uses 2 main blocks for feature extraction:

| Layer | Type | Parameters / Output Shape | Activation |
| --- | --- | --- | --- |
| **Input** | Image | `(28, 28, 1)` | - |
| **Block 1** | Conv2D | 32 filters, 3x3 kernel | ReLU |
|  | MaxPooling2D | 2x2 pool size | - |
|  | Batch Normalization | - | - |
| **Block 2** | Conv2D | 64 filters, 3x3 kernel | ReLU |
|  | MaxPooling2D | 2x2 pool size | - |
|  | Batch Normalization | - | - |
| **Flatten** | Flatten | 1D Vector | - |
| **Dense Head** | Dense | 128 neurons | ReLU |
|  | Dropout | Rate = 0.5 (To prevent overfitting) | - |
| **Output** | Dense | 10 neurons (Digits 0-9) | Softmax |

## 📊 Key Results and Analysis

* **CNN vs MLP Comparison:** Because the convolution blocks better learn spatial features and edges, the CNN model demonstrated noticeably higher accuracy and less tendency to overfit compared to the MLP.
* **Most Confused Digits (Confusion Matrix Analysis):** (Sample note) The most frequently confused digits by the model are the **4 and 9** pair. Hypothesis: When the top part of a handwritten "4" is joined, it visually closely resembles the circular shape of a "9".
* **Real-World Custom Image Test:** 5 digits written on real paper and photographed with a phone were first resized to 28x28 pixels, converted to Grayscale, and their colors were inverted (white digit on a black background, like MNIST). The model demonstrated high predictive power on these images.
