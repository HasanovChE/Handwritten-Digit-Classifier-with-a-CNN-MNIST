from pathlib import Path
from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="MNIST Digit Classifier API")

Instrumentator().instrument(app).expose(app)

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "tensorflow_cnn.keras"
)

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. "
        "Generate or download models/tensorflow_cnn.keras before starting the API."
    )

model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image):
    image = image.convert("L")
    image = image.resize((28, 28))
    image = ImageOps.invert(image)
    image = np.array(image, dtype=np.float32) / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

@app.get("/")
def root():
    return {"message": "MNIST CNN Digit Classifier"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    processed = preprocess_image(image)
    
    probabilities = model.predict(processed, verbose=0)[0]
    digit = int(np.argmax(probabilities))
    confidence = float(np.max(probabilities))
    
    return {
        "digit": digit,
        "confidence": confidence
    }