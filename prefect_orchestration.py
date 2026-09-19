# prefect_orchestration.py
from prefect import flow, task
import subprocess

@task(name="Run Model Retraining")
def run_retraining():
    print("🚀 Step 1: Model retraining begins...")
    subprocess.run(["python", "retraining_pipeline.py"], check=True)

@task(name="Validate Model Quality Gate")
def run_quality_gate():
    print("🔍 Step 2: Quality Gate check is being performed...")
    subprocess.run(["python", "validate_model_gate.py"], check=True)

@task(name="Convert to ONNX")
def convert_onnx():
    print("⚡ Step 3: The model is converted to ONNX format...")
    subprocess.run(["python", "convert_to_onnx.py"], check=True)

@flow(name="MNIST End-to-End MLOps Pipeline")
def mnist_mlops_pipeline():
    run_retraining()
    run_quality_gate()
    convert_onnx()
    print("✨ The entire MLOps pipeline completed successfully!")

if __name__ == "__main__":
    mnist_mlops_pipeline()